import numpy as np
import matplotlib.pyplot as plt

import netCDF4
import cartopy.crs as ccrs

class DataGrid(object):
    
    @classmethod
    def from_nc4(kls, file, varnames, latname = 'latitude', lonname = 'longitude', **kwargs):
        
        with netCDF4.Dataset(file) as rgrp:
            return kls.from_points(np.asarray(rgrp.variables[latname]),
                                   np.asarray(rgrp.variables[lonname]),
                                   dict((varname, np.asarray(rgrp.variables[varname])) for varname in varnames),
                                   **kwargs)       
    
    
    @classmethod
    def from_points(kls, lat, lon, data_arrs, pixel_size = np.array([0.5, 0.5]), dim_mul_of = [8, 8], **kwargs):
        """Construct a grid from a list of coordinates and data points.
        
        ASSUMES that there is only one data point in each pixel; chooses the last
        given value for a pixel if more than one.
        """
        
        mins = np.array([np.min(lat), np.min(lon)])
        maxes = np.array([np.max(lat), np.max(lon)])
        grid_dim = np.ceil((maxes - mins) / pixel_size)
        grid_dim = dim_mul_of * np.ceil(grid_dim / dim_mul_of)
        
        grid_lat, grid_lon = np.mgrid[0:grid_dim[0], 0:grid_dim[1]]
        grid_lat *= pixel_size[0]
        grid_lon *= pixel_size[1]

        grid_lat += mins[0] - 0.5 * pixel_size[0]
        grid_lon += mins[1] - 0.5 * pixel_size[1]
        
        grids = {}
        
        for varname in data_arrs:
            grids[varname] = np.full(grid_lat.shape, np.nan, dtype = np.float)
            grids[varname][np.floor((lat - mins[0]) / pixel_size[0]).astype(int), 
                           np.floor((lon - mins[1]) / pixel_size[1]).astype(int)] = data_arrs[varname]
            
        grids['n_samples'] = (~np.isnan(grids[varname])).astype(int)

        out = kls(grid_lat, grid_lon, grids, pixel_size, **kwargs)
    
        return out
    
    
    def __init__(self, lat_grid, lon_grid, grids, pixel_size, pretty_names = {}):
        self.grid_lat = lat_grid
        self.grid_lon = lon_grid
        self.data_grids = grids
        self.grid_dim = lat_grid.shape
        self.pixel_size = pixel_size
        
        self.pretty_names = pretty_names
       
        
    def __getattr__(self, varname):
        return self.data_grids[varname]
    
        
    def plot(self, to_show = True, title = None, figsize = (8, 6), cmaps = {}):
        
        cmaps_real = {'n_samples' : 'Blues'}
        cmaps_real.update(cmaps)
        
        if to_show is True:
            to_show = self.data_grids.keys()
        
        fig, axs = plt.subplots(
                        int(np.ceil(len(to_show) / 2)), 2,
                        subplot_kw={'projection' : ccrs.PlateCarree()}, 
                        figsize = figsize)
        
        axs = axs.flatten()
        
        for i, varname in enumerate(to_show):
            im = axs[i].pcolormesh(self.grid_lon,
                                   self.grid_lat, 
                                   self.data_grids[varname], 
                                   transform = ccrs.PlateCarree(),
                                   cmap = cmaps_real.get(varname, 'plasma'))
            
            axs[i].coastlines()   
            axs[i].set_title(self.pretty_names.get(varname, varname))
            
            fig.colorbar(im, ax = axs[i], fraction = 0.04)
            
        if i < len(axs) - 1:
            axs[-1].set_visible(False)
        
        if title is None:
            title = "Grid Data"
            
        fig.suptitle("%s (%.1fkm x %.1fkm pixels, avg. %i samples/bin)" % (
            title,
            self.pixel_size[0], self.pixel_size[1],
            np.mean(self.n_samples[self.n_samples != 0])
        ))
        
        return fig    
        
        
    def downscale(self, 
                  downscale_coeffs = (8, 8),
                  downsampling_functions = {}):
        
        dsfuncs = {'n_samples' : np.nansum}
        dsfuncs.update(downsampling_functions)
        
        new_shape = np.divide(self.grid_dim, downscale_coeffs).astype(int)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message = "Mean of empty slice")
            
            grids_downscaled = {}
            
            for varname in self.data_grids:
                grids_downscaled[varname] = DataGrid._rebin(self.data_grids[varname], 
                                                            new_shape,
                                                            func = dsfuncs.get(varname, np.nanmean))
            

        downscaled_pixel_size = np.multiply(self.pixel_size, downscale_coeffs)
        
        grid_lat_down = self.grid_lat[::downscale_coeffs[0], ::downscale_coeffs[1]]
        grid_lon_down = self.grid_lon[::downscale_coeffs[0], ::downscale_coeffs[1]]
        
        out =  type(self)(grid_lat_down, 
                          grid_lon_down, 
                          grids_downscaled,
                          pixel_size = downscaled_pixel_size)
        
        return out
        
        
    # https://scipython.com/blog/binning-a-2d-array-in-numpy/
    @staticmethod
    def _rebin(arr, new_shape, func = np.nanmean):
        assert arr.shape[0] % new_shape[0] == 0, "%s doesn't divide %s" % (new_shape[0], arr.shape[0])
        assert arr.shape[1] % new_shape[1] == 0, "%s doens't divide %s" % (new_shape[1], arr.shape[1])

        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return func(func(arr.reshape(shape), axis = -1), axis = 1)
    
    # -- Util Funcs --
    
    def copy(self):
        new_grids = dict((varname, self.data_grids[varname].copy()) for varname in self.data_grids)
        
        return type(self)(self.grid_lat.copy(),
                          self.grid_lon.copy(),
                          new_grids,
                          pixel_size = self.pixel_size)
    
    def __getitem__(self, varname):
        return self.data_grids[varname]
    
    # -- Operators --
    
    def _check_compatible(self, other, all_vars = True, strict = False):
        
        if isinstance(other, DataGrid):
        
            assert self.pixel_size == other.pixel_size, "Incompatible pixel sizes %s and %s; try downscaling" % (self.pixel_size, other.pixel_size)

            if strict:
                assert np.all(self.grid_lat == other.grid_lat), "Incomaptible latitude grid"
                assert np.all(self.grid_lon == other.grid_lon), "Incomaptible longitude grid"

            if all_vars:
                assert set(self.data_grids.keys()) == set(other.data_grids.keys())
                
        else:
            raise TypeError("DataGrid cannot do arithmetic with ")
    
    def _do_grid_func(self, other, func):
        self._check_compatible(self, other)
        
        for varname in self.data_grids:
            grid = self.data_grids[varname]
            hasdata = ~np.isnan(grid)
            getattr(grid[hasdata], func)(other[varname][hasdata])
            grid[~hasdata] = other[varname][~hasdata]
    
    def __iadd__(self, other):
        return self._do_grid_func(self, other, "__iadd__")
    def __isub__(self, other):
        return self._do_grid_func(self, other, "__isub__")
    def __imul__(self, other):
        return self._do_grid_func(self, other, "__imul__")
    def __idiv__(self, other):
        return self._do_grid_func(self, other, "__idiv__")
    
    def __add__(self, other):
        out = self.copy()
        out.__iadd__(other)
        return out
    def __sub__(self, other):
        out = self.copy()
        out.__isub__(other)
        return out
    def __mul__(self, other):
        out = self.copy()
        out.__imul__(other)
        return out
    def __div__(self, other):
        out = self.copy()
        out.__idiv__(other)
        return out
            
        