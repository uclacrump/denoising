import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import simpledialog, messagebox


def import_data(file_path):
    with open(file_path, 'rb') as file:
        return np.fromfile(file, dtype=np.float32)


def plot_slices(slice_index):
    global vol

    slices = vol[:, :, slice_index, :]

    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    axes = axes.ravel()

    for idx, slice_img in enumerate(slices):
        axes[idx].imshow(slice_img, cmap='gray', clim=(slice_img.min(), slice_img.max()))
        axes[idx].set_title(f"Slice {idx + 1}")
        axes[idx].axis('off')

    plt.tight_layout()

    return fig


def update_plot():
    global canvas

    slice_index = simpledialog.askinteger("Input", "Enter slice index:", minvalue=0, maxvalue=vol.shape[2] - 1)

    if slice_index is None:  # In case the user presses cancel
        return

    fig = plot_slices(slice_index)

    # Update canvas with the new figure
    canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# Import data and reshape
file_path = "amide_export_t58783_feldkamp"
data = import_data(file_path)
vol = data.reshape((-1, 200, 380, 380))

# GUI components
root = tk.Tk()
root.title("Slice Viewer")

try:
    # Display initial plot with default slice index 144
    canvas = FigureCanvasTkAgg(plot_slices(144), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    btn_update = tk.Button(root, text="Update Slice Index", command=update_plot)
    btn_update.pack(side=tk.BOTTOM)

    

    root.mainloop()
except Exception as e:
    messagebox.showerror("Error", str(e))
