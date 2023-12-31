import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import pdb

colormaps = [
    'gray',
    # Perceptually Uniform Sequential
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',

    # Sequential
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',

    # Sequential (2)
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',

    # Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',

    # Cyclic
    'twilight', 'twilight_shifted', 'hsv', 'hsv_r',

    # Qualitative
    'Pastel1', 'Pastel2', 'Paired', 'Accent',
    'Dark2', 'Set1', 'Set2', 'Set3',
    
    # Miscellaneous
    'tab10', 'tab20', 'tab20b', 'tab20c',
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    'cubehelix', 'brg', 'gist_rainbow', 'rainbow',
    'jet', 'turbo', 'nipy_spectral', 'gist_ncar'
]

def import_data(file_path):
    with open(file_path, 'rb') as file:
        return np.fromfile(file, dtype=np.float32)

def update_display(*args):
    cmap_choice = cmap_var.get()
    clim_min = float(clim_min_var.get())
    clim_max = float(clim_max_var.get())
    
    for idx, slice_img in enumerate(slices):
        axes[idx].imshow(slice_img, cmap=cmap_choice, clim=(clim_min, clim_max))
        axes[idx].set_title(f"Slice {idx + 1}")
        axes[idx].axis('off')
    
    canvas.draw()

file_path = "amide_export_t58783_feldkamp"
data = import_data(file_path)
vol = data.reshape((-1, 200, 380, 380))

slices = vol[:, :, 144, :]

root = tk.Tk()
root.title("Image Viewer")

mainframe = ttk.Frame(root, padding="1")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

cmap_var = tk.StringVar()
ttk.Label(mainframe, text="Color map:").grid(column=1, row=1, sticky=tk.W)

cmap_option = ttk.OptionMenu(mainframe, cmap_var, "gray", *colormaps, command=update_display)
# cmap_option = ttk.OptionMenu(mainframe, cmap_var, "gray", "gray", "bone", "hot", "cool", command=update_display)
cmap_option.grid(column=2, row=1, sticky=(tk.W, tk.E))

clim_min_var = tk.StringVar()
ttk.Label(mainframe, text="CLim min:").grid(column=1, row=2, sticky=tk.W)
clim_min_entry = ttk.Entry(mainframe, textvariable=clim_min_var)
clim_min_entry.grid(column=2, row=2, sticky=(tk.W, tk.E))
clim_min_var.set(str(int(np.min(slices))))

clim_max_var = tk.StringVar()
ttk.Label(mainframe, text="CLim max:").grid(column=1, row=3, sticky=tk.W)
clim_max_entry = ttk.Entry(mainframe, textvariable=clim_max_var)
clim_max_entry.grid(column=2, row=3, sticky=(tk.W, tk.E))
clim_max_var.set(str(int(np.max(slices))))

# Setup figure and canvas
fig, axes = plt.subplots(3, 4, figsize=(5.8, 3.73))
axes = axes.ravel()
canvas = FigureCanvasTkAgg(fig, master=mainframe)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(column=3, row=1, rowspan=3)

update_display()

root.mainloop()
