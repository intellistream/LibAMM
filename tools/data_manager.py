"""
LibAMM Data Manager
Manages data paths and downloads for LibAMM benchmarks
"""
import os
from pathlib import Path
import subprocess
import sys


class LibAMMDataManager:
    """Manages LibAMM data paths and setup"""
    
    def __init__(self, data_root=None):
        """
        Initialize data manager
        
        Args:
            data_root: Path to SAGE data directory. If None, will auto-detect.
        """
        self.data_root = self._find_data_root(data_root)
        self.libamm_data = self.data_root / "libamm-benchmark" if self.data_root else None
        
    def _find_data_root(self, data_root=None):
        """Find SAGE data root directory"""
        if data_root:
            return Path(data_root)
            
        # Check environment variable
        if "SAGE_DATA_ROOT" in os.environ:
            return Path(os.environ["SAGE_DATA_ROOT"])
            
        # Auto-detect relative to this file
        libamm_dir = Path(__file__).parent.parent
        sage_data = libamm_dir / "../../../../../sage-benchmark/src/sage/data"
        if sage_data.exists():
            return sage_data.resolve()
            
        return None
        
    def setup_links(self):
        """Create symbolic links to data directories"""
        if not self.libamm_data or not self.libamm_data.exists():
            print(f"❌ Error: LibAMM data not found at {self.libamm_data}")
            print("Please ensure sage-benchmark is installed or set SAGE_DATA_ROOT")
            return False
            
        libamm_root = Path(__file__).parent.parent
        
        # Create directories
        (libamm_root / "benchmark").mkdir(exist_ok=True)
        (libamm_root / "test/torchscripts/VQ").mkdir(parents=True, exist_ok=True)
        
        links = []
        
        # Link models
        models_src = self.libamm_data / "models"
        models_dst = libamm_root / "benchmark/models"
        if models_src.exists():
            self._create_symlink(models_src, models_dst)
            links.append(f"benchmark/models -> {models_src}")
            
        # Link test data
        test_data_src = self.libamm_data / "test-data"
        test_data_dst = libamm_root / "test/torchscripts/VQ/data"
        if test_data_src.exists():
            self._create_symlink(test_data_src, test_data_dst)
            links.append(f"test/torchscripts/VQ/data -> {test_data_src}")
            
            # Also link individual files for compatibility
            for file in test_data_src.glob("*.txt"):
                dst = libamm_root / "test/torchscripts/VQ" / file.name
                self._create_symlink(file, dst)
                
        # Link datasets
        datasets_src = self.libamm_data / "datasets"
        datasets_dst = libamm_root / "benchmark/datasets"
        if datasets_src.exists():
            self._create_symlink(datasets_src, datasets_dst)
            links.append(f"benchmark/datasets -> {datasets_src}")
            
        print("✅ LibAMM data setup complete!")
        for link in links:
            print(f"  ✓ {link}")
        return True
        
    def _create_symlink(self, src, dst):
        """Create symbolic link, removing existing link/file if needed"""
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            print(f"⚠️  Warning: {dst} exists and is not a symlink, skipping")
            return
        dst.symlink_to(src)
        
    def get_dataset_path(self, dataset_name):
        """
        Get path to a dataset
        
        Args:
            dataset_name: Name of dataset (e.g., "QCD/qcda_small.mtx")
            
        Returns:
            Path to dataset file
        """
        if not self.libamm_data:
            raise RuntimeError("LibAMM data root not found")
        return self.libamm_data / "datasets" / dataset_name
        
    def get_model_path(self, model_name):
        """Get path to a model file"""
        if not self.libamm_data:
            raise RuntimeError("LibAMM data root not found")
        return self.libamm_data / "models" / model_name


def setup_libamm_data():
    """Setup LibAMM data links - called during installation"""
    manager = LibAMMDataManager()
    success = manager.setup_links()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(setup_libamm_data())
