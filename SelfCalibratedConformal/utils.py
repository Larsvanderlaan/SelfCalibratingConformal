def make_grid(values, num_bin, values_to_add = None, binning_method = "quantile"):
    unique_values = list(set(values))
    unique_values.sort()
    if len(unique_values) <= num_bin:
      return unique_values
    if binning_method == "quantile":
      qs = np.linspace(0, 1, num = num_bin + 1)
      grid_values = set(np.quantile(values, qs, method = 'inverted_cdf')) 
      if(values_to_add != None):
        grid_values.union(set(values_to_add))
      grid_values = list(grid_values) 
      grid_values.sort()
    elif binning_method == "fixed":
      grid_values = np.linspace(min(values), max(values), num = num_bin + 1)
    return grid_values
  
def match_grid_value(values, grid, return_index = False, all_inside = False):
    bin_index = np.searchsorted(grid, values, side = 'right')
    if(all_inside):
      bin_index[bin_index == len(grid)] = len(grid) - 1
    bin_index = [i if i == 0 else i-1 for i in bin_index]
    if not return_index:
      values_discrete = [grid[i] for i in bin_index]
      return values_discrete
    else:
      return bin_index
