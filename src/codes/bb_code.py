from typing import Tuple, List, Dict, Any
import numpy as np

class BBCodeCircuit:
    """
    Constructs the syndrome extraction circuit for a Bivariate Bicycle code.
    
    The circuit structure is deduced from the component matrices A1, A2, A3, B1, B2, B3.
    For BB codes:
    - Hx = [A | B] where A = A1 + A2 + A3 and B = B1 + B2 + B3
    """
    
    def __init__(
        self, 
        Hx: np.ndarray, 
        Hz: np.ndarray, 
        num_cycles: int = 12,
        ell: int = None,
        m: int = None,
        a_x_powers: np.ndarray = None,
        a_y_powers: np.ndarray = None,
        b_y_powers: np.ndarray = None,
        b_x_powers: np.ndarray = None,
    ):
        self.Hx = np.asarray(Hx, dtype=int)
        self.Hz = np.asarray(Hz, dtype=int)
        self.num_cycles = num_cycles
        
        self.m_checks, self.n = Hx.shape
        self.n2 = self.n // 2
        
        assert self.m_checks == self.n2, f"Expected square blocks: m={self.m_checks}, n2={self.n2}"
        
        self.ell = ell
        self.m_dim = m
        self.a_x_powers = a_x_powers if a_x_powers is not None else []
        self.a_y_powers = a_y_powers if a_y_powers is not None else []
        self.b_y_powers = b_y_powers if b_y_powers is not None else []
        self.b_x_powers = b_x_powers if b_x_powers is not None else []
        
        self.has_component_params = ell is not None and m is not None
        if self.has_component_params:
            self._build_component_matrices()
        
        self._setup_qubit_ordering()
        self._compute_neighbors()
        self._build_cnot_schedule()
        self._build_cycle()
    
    def _build_component_matrices(self):
        ell = self.ell
        m = self.m_dim
        I_ell = np.eye(ell, dtype=int)
        I_m = np.eye(m, dtype=int)
        
        self.A_components = []
        for p in self.a_x_powers:
            self.A_components.append(np.kron(np.roll(I_ell, p, axis=1), I_m))
        for p in self.a_y_powers:
            self.A_components.append(np.kron(I_ell, np.roll(I_m, p, axis=1)))
        
        self.B_components = []
        for p in self.b_y_powers:
            self.B_components.append(np.kron(I_ell, np.roll(I_m, p, axis=1)))
        for p in self.b_x_powers:
            self.B_components.append(np.kron(np.roll(I_ell, p, axis=1), I_m))
        
        while len(self.A_components) < 3:
            self.A_components.append(np.zeros((self.n2, self.n2), dtype=int))
        while len(self.B_components) < 3:
            self.B_components.append(np.zeros((self.n2, self.n2), dtype=int))
        
    def _setup_qubit_ordering(self):
        self.lin_order = {}
        self.data_qubits = []
        self.Xchecks = []
        self.Zchecks = []
        
        cnt = 0
        for i in range(self.n2):
            node = ('Xcheck', i)
            self.Xchecks.append(node)
            self.lin_order[node] = cnt
            cnt += 1
        
        for i in range(self.n2):
            node = ('data_left', i)
            self.data_qubits.append(node)
            self.lin_order[node] = cnt
            cnt += 1
            
        for i in range(self.n2):
            node = ('data_right', i)
            self.data_qubits.append(node)
            self.lin_order[node] = cnt
            cnt += 1
            
        for i in range(self.n2):
            node = ('Zcheck', i)
            self.Zchecks.append(node)
            self.lin_order[node] = cnt
            cnt += 1
            
        self.total_qubits = cnt
        
    def _compute_neighbors(self):
        self.nbs = {}
        if self.has_component_params:
            A1, A2, A3 = self.A_components[0], self.A_components[1], self.A_components[2]
            B1, B2, B3 = self.B_components[0], self.B_components[1], self.B_components[2]
            
            for i in range(self.n2):
                check = ('Xcheck', i)
                self.nbs[(check, 0)] = ('data_left', np.nonzero(A1[i, :])[0][0] if np.any(A1[i, :]) else 0)
                self.nbs[(check, 1)] = ('data_left', np.nonzero(A2[i, :])[0][0] if np.any(A2[i, :]) else 0)
                self.nbs[(check, 2)] = ('data_left', np.nonzero(A3[i, :])[0][0] if np.any(A3[i, :]) else 0)
                self.nbs[(check, 3)] = ('data_right', np.nonzero(B1[i, :])[0][0] if np.any(B1[i, :]) else 0)
                self.nbs[(check, 4)] = ('data_right', np.nonzero(B2[i, :])[0][0] if np.any(B2[i, :]) else 0)
                self.nbs[(check, 5)] = ('data_right', np.nonzero(B3[i, :])[0][0] if np.any(B3[i, :]) else 0)
            
            B1T, B2T, B3T = B1.T, B2.T, B3.T
            A1T, A2T, A3T = A1.T, A2.T, A3.T
            
            for i in range(self.n2):
                check = ('Zcheck', i)
                self.nbs[(check, 0)] = ('data_left', np.nonzero(B1T[i, :])[0][0] if np.any(B1T[i, :]) else 0)
                self.nbs[(check, 1)] = ('data_left', np.nonzero(B2T[i, :])[0][0] if np.any(B2T[i, :]) else 0)
                self.nbs[(check, 2)] = ('data_left', np.nonzero(B3T[i, :])[0][0] if np.any(B3T[i, :]) else 0)
                self.nbs[(check, 3)] = ('data_right', np.nonzero(A1T[i, :])[0][0] if np.any(A1T[i, :]) else 0)
                self.nbs[(check, 4)] = ('data_right', np.nonzero(A2T[i, :])[0][0] if np.any(A2T[i, :]) else 0)
                self.nbs[(check, 5)] = ('data_right', np.nonzero(A3T[i, :])[0][0] if np.any(A3T[i, :]) else 0)
        else:
            for i in range(self.n2):
                check = ('Xcheck', i)
                row = self.Hx[i, :]
                left_indices = np.nonzero(row[:self.n2])[0]
                right_indices = np.nonzero(row[self.n2:])[0]
                for j, idx in enumerate(left_indices[:3]):
                    self.nbs[(check, j)] = ('data_left', idx)
                for j, idx in enumerate(right_indices[:3]):
                    self.nbs[(check, 3 + j)] = ('data_right', idx)
                    
            for i in range(self.n2):
                check = ('Zcheck', i)
                row = self.Hz[i, :]
                left_indices = np.nonzero(row[:self.n2])[0]
                right_indices = np.nonzero(row[self.n2:])[0]
                for j, idx in enumerate(left_indices[:3]):
                    self.nbs[(check, j)] = ('data_left', idx)
                for j, idx in enumerate(right_indices[:3]):
                    self.nbs[(check, 3 + j)] = ('data_right', idx)
                
    def _build_cnot_schedule(self):
        self.schedule_X = ['idle', 1, 4, 3, 5, 0, 2, 'idle']
        self.schedule_Z = [3, 5, 0, 1, 2, 4, 'idle', 'idle']
        
    def _build_cycle(self):
        self.cycle = []
        num_rounds = 8
        for t in range(num_rounds):
            ops_this_round = []
            data_qubits_cnoted = set()
            if t == 0:
                for q in self.Xchecks:
                    ops_this_round.append(('PrepX', q))
            if self.schedule_X[t] != 'idle':
                direction = self.schedule_X[t]
                for control in self.Xchecks:
                    target = self.nbs[(control, direction)]
                    ops_this_round.append(('CNOT', control, target))
                    data_qubits_cnoted.add(target)
            if self.schedule_Z[t] != 'idle':
                direction = self.schedule_Z[t]
                for target in self.Zchecks:
                    control = self.nbs[(target, direction)]
                    ops_this_round.append(('CNOT', control, target))
                    data_qubits_cnoted.add(control)
            for q in self.data_qubits:
                if q not in data_qubits_cnoted:
                    ops_this_round.append(('IDLE', q))
            if t == 6:
                for q in self.Zchecks:
                    ops_this_round.append(('MeasZ', q))
            if t == 7:
                for q in self.Xchecks:
                    ops_this_round.append(('MeasX', q))
                for q in self.Zchecks:
                    ops_this_round.append(('PrepZ', q))
            self.cycle.extend(ops_this_round)
            
    def get_full_circuit(self) -> List[Tuple]:
        return self.cycle * self.num_cycles
    
    def get_circuit_with_final_measurements(self) -> List[Tuple]:
        noisy_circuit = self.cycle * self.num_cycles
        noiseless_suffix = self.cycle * 2
        return noisy_circuit, noiseless_suffix
