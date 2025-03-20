import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from util import make_time_sequence
import torch


def test_and_plot_graphs(
    model,
    device,
    x_range=[0, 1], 
    y_range=[0, 1], 
    timestamp=1, 
    nop=200, 
    save_images=True,
    k_steps_foward=None,
    step_size=1e-4
):      
    Nx = nop
    Ny = nop


    # Build plot grid
    plot_grid = np.mgrid[x_range[0] : x_range[1] : Nx * 1j, y_range[0] : y_range[1] : Ny * 1j]
    X, Y = plot_grid
    

    points = np.vstack((X.ravel(), Y.ravel(), np.ones(X.size) * timestamp))


    # Generate pseudo-sequences for PINNsFormer
    if not k_steps_foward is None: 
        points = make_time_sequence(points, k_steps_foward, step_size)
    
    points = torch.tensor(points.T, dtype=torch.float32, requires_grad=True).to(device)


    if k_steps_foward is None:
        x_test, y_test, t_test = points[:,0:1], points[:,1:2], points[:,2:3]
    else:
        x_test, y_test, t_test = points[:,:,0:1], points[:,:,1:2], points[:,:,2:3]


    with torch.no_grad():
        predicted = model(x_test, y_test, t_test)


    if k_steps_foward is None:
        u = predicted[:, 0:1].reshape((Nx, Ny))
        v = predicted[:, 1:2].reshape((Nx, Ny))
        p = predicted[:, 2:3].reshape((Nx, Ny))
    else:
        u = predicted[:, 0, 0:1].reshape((Nx, Ny))
        v = predicted[:, 0, 1:2].reshape((Nx, Ny))
        p = predicted[:, 0, 2:3].reshape((Nx, Ny))

    
    matrix_U = np.fliplr(u).T
    matrix_V = np.fliplr(v).T
    matrix_p = np.fliplr(p).T

    # For image saving
    model_name = model.__class__.__name__
    
    # Plot u velocity heatmap
    fig, (ax1) = plt.subplots(1, 1)
    im1 = ax1.imshow(matrix_U, cmap='jet')
    ax1.contour(Y * Ny, X * Nx, matrix_U, levels=20, colors='white', linewidths=1.2)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    
    if save_images:
        plt.savefig(f"images/{model_name}-u.png")

    
    # Plot v velocity heatmap    
    fig, (ax2) = plt.subplots(1, 1)
    im2 = ax2.imshow(matrix_V, cmap='jet')
    ax2.contour(Y * Ny, X * Nx, matrix_V, levels=20, colors='white', linewidths=1.2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    if save_images:
        plt.savefig(f"images/{model_name}-v.png")

    
    # Plot preassure heatmap
    fig, (ax3) = plt.subplots(1, 1)
    im3 = ax3.imshow(matrix_p, cmap='jet')
    ax3.contour(Y * Ny, X * Nx, matrix_p, levels=20, colors='white', linewidths=1.2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    if save_images:
        plt.savefig(f"images/{model_name}-p.png")

    
    # Compare preditected v velocity against Ghia et al. (1982) bechmark values 
    
    # Ghia, U. K. N. G., Ghia, K. N., & Shin, C. T. (1982)
    #https://gist.github.com/ivan-pi/caa6c6737d36a9140fbcf2ea59c78b3c#file-ghiav-txt
    ghiav_bechmark = np.loadtxt("data/ghiav.txt")
    
    ref_x =  ghiav_bechmark[:, 0:1] # x coord
    ref_v =  ghiav_bechmark[:, 1:2] # Re = 100
    
    fig, (ax4) = plt.subplots(1, 1)
    ax4.plot(ref_x, ref_v, 'ro', plot_grid[1,int(nop/2),:], v[int(nop/2),:])

    if save_images:
        plt.savefig(f"images/{model_name}-benchmark.png")

    
    plt.show()
    