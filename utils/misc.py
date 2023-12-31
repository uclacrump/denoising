import numpy as np

def calculate_cropped_index_of_interest(index_of_interest, dim_of_interest, original_shape, new_shape):
    _, old_depth, old_rows, old_cols = original_shape
    _, new_depth, new_rows, new_cols = new_shape
    
    start_depth = (old_depth - new_depth) // 2
    start_row = (old_rows - new_rows) // 2
    start_col = (old_cols - new_cols) // 2
    
    new_index = None
    
    if dim_of_interest == 'depth':
        new_index = index_of_interest - start_depth
    elif dim_of_interest == 'rows':
        new_index = index_of_interest - start_row
    elif dim_of_interest == 'cols':
        new_index = index_of_interest - start_col
    
    return new_index

def calculate_downsampled_index_of_interest(old_shape, new_shape, axis_of_interest, index_of_interest):
    old_depth, old_rows, old_cols = old_shape
    new_depth, new_rows, new_cols = new_shape
    
    depth_factor = old_depth // new_depth
    row_factor = old_rows // new_rows
    col_factor = old_cols // new_cols
    
    new_index = None
    
    if axis_of_interest == 'depth':
        new_index = index_of_interest // depth_factor
    elif axis_of_interest == 'rows':
        new_index = index_of_interest // row_factor
    elif axis_of_interest == 'cols':
        new_index = index_of_interest // col_factor
    
    if new_index is not None:
        return new_index
    else:
        raise ValueError("Invalid axis_of_interest. Choose from 'depth', 'rows', or 'cols'.")
    

def import_data(file_path):
    with open(file_path, 'rb') as file:
        return np.fromfile(file, dtype=np.float32)

def export_data(data, file_path):
    with open(file_path, 'wb') as file:
        data.astype(np.float32).tofile(file)

def load_images(file_path, shape):
    data = import_data(file_path)
    vol = data.reshape(shape)
    print(f"Volume shape: {vol.shape}")
    return vol