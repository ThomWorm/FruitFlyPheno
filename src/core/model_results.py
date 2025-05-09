import xarray as xr
#scaffold
class ModelResults:
    def __init__(self, recent_data: xr.Dataset):
        """Initialize with your spatial_wrapper output"""
        self.data = recent_data
        self._validate()
    
    def _validate(self):
        """Ensure required variables exist"""
        required_vars = {'days_to_f3_completion', 'incomplete_development', 'missing_data'}
        assert all(var in self.data for var in required_vars)
    
    def has_incomplete_areas(self) -> bool:
        """Replace checks for -1s"""
        return bool(self.data['incomplete_development'].any())
    
    def get_incomplete_mask(self) -> xr.DataArray:
        """Clean access to incomplete areas"""
        return self.data['incomplete_development']
    
    # Add analysis methods as needed
    def completion_rate(self) -> xr.DataArray:
        """Percentage completion by location"""
        return (self.data['days_to_f3_completion'] > 0).mean(dim='time')