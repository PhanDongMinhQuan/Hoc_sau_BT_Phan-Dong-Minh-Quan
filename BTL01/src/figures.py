"""Figures and tables are derived from saved measurements, not typed results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .data import load_split, gaussian, normalize_inputs
from .training import load_model
from .model import predict

COLORS={"train":"#245C8C","validation":"#C06B24","test":"#2A856E"}


def create_figures(root):
    root=Path(root)
    out=root/"figures"
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold","savefig.dpi":180,"figure.facecolor":"white"})
    def save(fig,name):
        # Matplotlib's 3D tight bounding box may omit the vertical z label.
        extra = [ax.zaxis.label for ax in fig.axes if hasattr(ax, "zaxis")]
        fig.savefig(out/name,bbox_inches="tight",bbox_extra_artists=extra or None,
                    pad_inches=0.15,facecolor="white")
        plt.close(fig)
    splits={name:load_split(root/"data",name) for name in ["train","validation","test"]}
    x,y=splits["train"]
    g1,g2=np.meshgrid(np.linspace(-2,2,121),np.linspace(-4,4,121))
    grid=np.column_stack([g1.ravel(),g2.ravel()]); z=gaussian(grid).reshape(g1.shape)
    fig=plt.figure(figsize=(10,4))
    a=fig.add_subplot(121,projection="3d");a.plot_surface(g1,g2,z,cmap="viridis",linewidth=0,rstride=3,cstride=3)
    a.set(title="Hàm Gaussian mục tiêu",xlabel="$x_1$",ylabel="$x_2$",zlabel="$f(x)$")
    b=fig.add_subplot(122,projection="3d");b.scatter(x[:500,0],x[:500,1],y[:500,0],c=y[:500,0],s=8,cmap="viridis")
    b.set(title="500 điểm đầu của tập train",xlabel="$x_1$",ylabel="$x_2$",zlabel="$y$")
    save(fig,"01_data_3d.png")
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for i,(values,label) in enumerate([(x[:,0],"$x_1$"),(x[:,1],"$x_2$"),(y[:,0],"$y$")]):
        axes[i].hist(values,bins=25,color=["#245C8C","#2A856E","#C06B24"][i],edgecolor="white",linewidth=.4)
        axes[i].set(xlabel=label,ylabel="Số mẫu",title=f"Phân bố {label}")
    fig.tight_layout();save(fig,"02_distributions.png")
    stats=[]
    for name,(sx,sy) in splits.items():
        for label,values in [("x1",sx[:,0]),("x2",sx[:,1]),("y",sy[:,0])]:
            stats.append({"split":name,"variable":label,"n":len(values),"min":values.min(),"max":values.max(),"mean":values.mean(),"std":values.std(ddof=1)})
    pd.DataFrame(stats).to_csv(root/"results/data_statistics.csv",index=False)
    summary=pd.read_csv(root/"results/summary.csv")
    for group in ["A","B"]:
        s=summary[summary.group==group]
        metrics=["mse","mae","rmse"] if group=="A" else ["mse","mae"]
        fig,axes=plt.subplots(1,len(metrics),figsize=(10,3.4))
        for ax,metric in zip(axes,metrics):
            for split in COLORS:
                ax.errorbar(s.value,s[f"{split}_{metric}_mean"],yerr=s[f"{split}_{metric}_std"],label=split.capitalize(),color=COLORS[split],marker="o",markersize=3,capsize=2,linewidth=1.2)
            ax.set(xlabel="Số mẫu train A" if group=="A" else "Số nơ-ron B",ylabel=metric.upper(),yscale="log")
            if group=="B":
                ax.set_xscale("log",base=2);ax.set_xticks([1,2,4,8,16,32,64],labels=[1,2,4,8,16,32,64])
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();save(fig,f"0{3 if group=='A' else 4}_experiment_{group}.png")
    s=summary[summary.group=="C"]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for split in COLORS:
        axes[0].errorbar(s.value,s[f"{split}_mse_mean"],yerr=s[f"{split}_mse_std"],label=split.capitalize(),color=COLORS[split],marker="o",capsize=3)
    axes[0].set(yscale="log",xlabel="Số tầng ẩn C",ylabel="MSE");axes[0].legend(fontsize=8)
    axes[1].errorbar(s.value,s.training_seconds_mean,yerr=s.training_seconds_std,color="#245C8C",marker="o",capsize=3)
    axes[1].set(xlabel="Số tầng ẩn C",ylabel="Thời gian huấn luyện (giây)")
    for ax in axes: ax.set_xticks(s.value);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,"05_experiment_C.png")
    runs=pd.read_csv(root/"results/all_runs.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for _,row in runs[(runs.group=="C")&(runs.seed==38)].iterrows():
        hist=pd.read_csv(root/"results/runs"/row.run_id/"history.csv")
        axes[0].plot(hist.epoch,hist.train_mse,label=f"C={row.value}",linewidth=1)
        axes[1].plot(hist.epoch,hist.validation_mse,label=f"C={row.value}",linewidth=1)
    for ax,label in zip(axes,["Train MSE","Validation MSE"]):
        ax.set(xlabel="Epoch",ylabel=label,yscale="log");ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,"06_depth_convergence.png")
    selection=json.loads((root/"results/selection.json").read_text(encoding="utf-8"))
    model,ckpt=load_model(root/"checkpoints/selected_model.pt")
    hist=pd.read_csv(root/"results/runs"/selection["representative_run_id"]/"history.csv")
    fig,ax=plt.subplots(figsize=(8,3.3))
    ax.plot(hist.epoch,hist.train_mse,label="Train",color=COLORS["train"])
    ax.plot(hist.epoch,hist.validation_mse,label="Validation",color=COLORS["validation"])
    ax.axvline(ckpt["best_epoch"],color="#64748B",ls="--",label=f'Checkpoint: epoch {ckpt["best_epoch"]}')
    ax.set(xlabel="Epoch",ylabel="MSE",yscale="log");ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,"07_selected_convergence.png")
    t=pd.read_csv(root/"results/test_predictions.csv")
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    axes[0].scatter(t.y_true,t.y_pred,s=18,color=COLORS["test"],alpha=.8)
    limit=max(t.y_true.max(),t.y_pred.max())*1.04
    axes[0].plot([0,limit],[0,limit],"--",color="#64748B",label="Dự đoán đúng")
    axes[0].set(xlabel="Giá trị thật y",ylabel="Dự đoán ŷ");axes[0].legend(fontsize=8)
    axes[1].scatter(t.y_true,t.error,s=18,color=COLORS["test"]);axes[1].axhline(0,color="#64748B",ls="--")
    axes[1].set(xlabel="Giá trị thật y",ylabel="Sai số ŷ − y")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();save(fig,"08_test_predictions.png")
    pred=predict(model,normalize_inputs(grid)).reshape(z.shape)
    fig=plt.figure(figsize=(10,4))
    for i,(values,title) in enumerate([(z,"Hàm thật"),(pred,f'MLP {selection["hidden"]}')],1):
        ax=fig.add_subplot(1,2,i,projection="3d")
        ax.plot_surface(g1,g2,values,cmap="viridis",vmin=0,vmax=z.max(),rstride=3,cstride=3)
        ax.set(title=title,xlabel="$x_1$",ylabel="$x_2$",zlabel="Mật độ",zlim=(-.015,.28))
    save(fig,"09_surface_comparison.png")
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    im=axes[0].imshow(abs(pred-z),extent=(-2,2,-4,4),origin="lower",aspect="auto",cmap="magma")
    axes[0].set(xlabel="$x_1$",ylabel="$x_2$",title="Sai số tuyệt đối trên lưới")
    fig.colorbar(im,ax=axes[0],label="|ŷ − y|")
    axes[1].hist(t.error,bins=15,color=COLORS["test"],edgecolor="white")
    axes[1].set(xlabel="Sai số trên 100 mẫu test",ylabel="Số mẫu",title="Phân bố phần dư")
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].ticklabel_format(axis="x",style="sci",scilimits=(0,0),useMathText=True)
    fig.tight_layout();save(fig,"10_spatial_error.png")
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for i in [0,1]:
        line=np.zeros((300,2)); line[:,i]=np.linspace([-2,-4][i],[2,4][i],300)
        axes[i].plot(line[:,i],gaussian(line),color=COLORS["train"],label="Hàm thật")
        axes[i].plot(line[:,i],predict(model,normalize_inputs(line)),color=COLORS["validation"],ls="--",label="MLP")
        axes[i].set(xlabel=f"$x_{i+1}$",ylabel="Mật độ",title="$x_2=0$" if i==0 else "$x_1=0$")
        axes[i].legend(fontsize=9);axes[i].grid(alpha=.2)
    fig.tight_layout();save(fig,"11_cross_sections.png")
    print(f"Saved {len(list(out.glob('*.png')))} figures to {out}")
