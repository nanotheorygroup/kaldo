from abc import ABC, abstractmethod
from enum import Enum

class AbstractForceConstantStorageFormat(ABC):

    def __init(self, ifcs, fmt_type, err_msg):
        try:
            self.order = fmt_type.order_map[len(ifcs.shape)]
        except KeyError:
            raise KeyError(err_msg)
        
        self.ifcs = ifc_tensor


    # Define dunder methods so this type acts like a raw numpy array
    #! MIGHT NEED OTHER ONES
    def __getitem__(self, index):
        return self.ifcs[index]

    def __setitem__(self, index, value):
        self.ifcs[index] = value

    def __len__(self):
        return len(self.ifcs)

    def __iter__(self):
        return iter(self.ifcs)

    def __next__(self):
        return next(self.ifcs)


class ReplicaForceConstantFormat(AbstractForceConstantStorageFormat):

    order_map = {
        6 : 2,
        9 : 3,
        11: 4,
    }

    def __init__(self, ifcs):
        err_msg = f"Replica IFC format has invalid shape: {ifcs.shape}." + 
                            "Expected something like (n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)"
        super().__init__(ifcs, ReplicaForceConstantFormat, err_msg)
        
    @abstractmethod
    def to_block_sparse_fmt(self) -> BlockSparseForceConstantFormat:
        pass

#! IM NOT SURE WHAT TO CALL THIS FORMAT
class BlockSparseForceConstantFormat(AbstractForceConstantStorageFormat):
    """
    In neighbor list format IFCs store all interactions between primitive atoms and supercell atoms.
    Second order IFCs have shape: (n_unit_atoms, n_sc, 3, 3)
    Third order IFCs have shape: (n_unit_atoms, n_sc, n_sc, 3, 3, 3)
    Fourth order IFCs have shape: (n_unit_atoms, n_sc, n_sc, n_sc, 3, 3, 3, 3)

    This is the default format used by TDEP, and naturally maps on to block sparse formats.
    """

    order_map = {
        4 : 2,
        6 : 3,
        8 : 4,
    }

    def __init__(self, ifcs):
        err_msg = f"Neighbor list IFC format has invalid shape: {ifcs.shape}." + 
                            "Expected something like (n_unit_atoms, n_sc, 3, 3)"
        super().__init__(ifcs, BlockSparseForceConstantFormat, err_msg)

    @abstractmethod
    def to_replica_fmt(self) -> ReplicaForceConstantFormat:
        pass

class AbstractForceConstantFormat(ABC):
    
    @abstractmethod
    def load_second_order(
        self,
        filename : str,
        supercell : tuple[int, int, int],
    ) -> SecondOrder:
        pass

    @abstractmethod
    def load_third_order(
        self,
        filename : str,
        supercell : tuple[int, int, int],
    ) -> ThirdOrder:
        pass

    @abstractmethod
    def load_fourth_order(
        self,
        filename : str,
        supercell : tuple[int, int, int],
    ) -> FourthOrder:
        pass    

    @abstractmethod
    def convert_storage_to(
        self,
        storage_fmt : AbstractForceConstantStorageFormat,
    ) -> AbstractForceConstantFormat:
        pass