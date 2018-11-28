# =============================================================================
# HTM.py - Hierarchical Temporal Memory Nupic Implementation
# Copyright (C) 2018  Humza Syed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

import numpy as np
from nupic.bindings.algorithms import SpatialPooler #CPP implementation
from nupic.algorithms.backtracking_tm_cpp import BacktrackingTMCPP as TemporalMemory # CPP implementation

def init_HTM(# SP inputs
             input_window, n_cols, pot_radius, pot_percent, inhibition, act_cols_area, stim_thresh, perm_dec_SP, perm_inc_SP, perm_thresh, boosting, random_seed,
             # TM inputs, only similarity is n_cols and random_seed
             n_cells_per_col, init_perm, min_thresh, new_syn_learning, perm_inc_TM, perm_dec_TM, age, decay, act_thresh, pam_l, max_seg_per_cell, max_syn_per_seg
             ):
    """
    Instantiations for Spatial Pooler and Temporal Memory for HTM model
    Parameters are described separately for the Spatial Pooler and Temporal Memory
    """

    """
    Initialization of SP instance
    reference: nupic/src/nupic/algorithms/spatial_pooler.py for class declaration
    each default is according to Nupic's default
    :param inputDimensions  (int)             : 1 dimensional input data for streaming data; default=(32,32)
    :param columnDimensions (int)             : the number of mini-columns; default=(64,64)=4096 mini-columns
    :param potentialRadius  (int)             : receptive field; default=16
    :param potentialPct     (float)           : percentage of inputs that mini-columns are connected to; default=0.5
    :param globalInhibition (bool)            : determines if winning columns are selected based off the global region or based off local regions; default=False
    :param numActiveColumnsPerInhArea (float) : at most these number of mini-columns need to be active per inhibition area; default=40
    :param stimulusThreshold (int)            : minimum number of synapess needed to be on for column to become active; default=0
    :param synPermInactiveDec (float)         : Hebbian permanence decrement value for synapses; default=0.008
    :param synPermActiveInc (float)           : Hebbian permanence increment value for synapses; default=0.05
    :param synPermConnected (float)           : permanence threshold for synapse to become formed; default=0.1
    :param boostStrength (float)              : boosting factor for encouraging inactive mini-columns; default=0.0
    :param seed (int)                         : random seed value for repeatability of experiments; default=-1
    """

    SP = SpatialPooler(inputDimensions=(input_window,), 
                       columnDimensions=(n_cols,), 
                       potentialRadius=pot_radius, 
                       potentialPct=pot_percent, 
                       globalInhibition=inhibition, 
                       numActiveColumnsPerInhArea=act_cols_area,
                       stimulusThreshold=stim_thresh,
                       synPermInactiveDec=perm_dec_SP,
                       synPermActiveInc=perm_inc_SP,
                       synPermConnected=perm_thresh,
                       boostStrength=boosting,
                       seed=random_seed
                       )

    """
    Initialization of TM instance
    reference: nupic/src/nupic/algorithms/temporal_memory.py
    each default is according to NuPic's defaults
    :param numberOfCols (int)          : the number of mini-columns; default=500
    :param cellsPerColumn (int)        : the number of cells per mini-column; default=10
    :param initialPerm (float)         : initial permanence values for newly created synapses; default=0.11
    :param minThreshold (int)          : minimum number of active synapses for a segment when searching for best-matching segments; default=8
    :param newSynapseCount (int)       : max number of synapses added to a segment during training; default=15
    :param permanenceInc (float)       : Hebbian permanence increment value for synapses; default=0.10
    :param permanenceDec (float)       : Hebbian permanence decrement value for synapses; default=0.10
    :param maxAge (int)                : number of iterations before globabl decay takes effect; default=100000
    :param globalDecay (float)         : value to decrease permanence when global decay occurs --> global decay is meant to remove synapses when perm value reaches 0; default=0.10
    :param activationThreshold (int)   : number of synapses that must be active to activate a segment; default=12
    :param seed (int)                  : random seed value for repeatability of experiments; default=42
    :param pamLength (int)             : number of time steps to remain in "Pay Attention Mode"; default=1
    :param maxSegmentsPerCell (int)    : the maximum number of segments allowed per cell; default=-1
    :param maxSynapsesPerSegment (int) : the maximum number of synapses per segment; default=-1
    """
    TM = TemporalMemory(numberOfCols=n_cols,
                        cellsPerColumn=n_cells_per_col,
                        initialPerm=init_perm,
                        minThreshold=min_thresh,
                        newSynapseCount=new_syn_learning,
                        permanenceInc=perm_inc_TM,
                        permanenceDec=perm_dec_TM,
                        maxAge=age,
                        globalDecay=decay,
                        activationThreshold=act_thresh,
                        seed=random_seed,
                        pamLength=pam_l,
                        maxSegmentsPerCell=max_seg_per_cell,
                        maxSynapsesPerSegment=max_syn_per_seg
                        )

    return HTM_Model(SP, TM)

class HTM_Model(object):
    """
    Class to manage Spatial Pooler and Temporal Memory
    """

    def __init__(self, SP, TM):
        self.SP = SP
        self.TM = TM
        self.activeCols = np.zeros(TM.numberOfCols, dtype="float32")
        self.predCols = np.zeros(TM.numberOfCols, dtype="float32")
        self.learn = True
    
    def get_SP(self):
        """
        Return instance of SP
        """

        return self.SP

    def get_TM(self):
        """
        Return instance of TM
        """
        
        return self.TM

    def get_activeCols(self):
        """
        Return the output of SP.
        """
        return self.activeCols.nonzero()[0]
    
    def get_predCols(self):
        """
        Return the output of TM.
        """
        return self.predCols.nonzero()[0]

    def stop_learning(self):
        """
        Turn learning off for both SP and TM
        """
        self.learn = False

    def start_learning(self):
        """
        Turn learning back on for both SP and TM
        """
        self.learn = True

    def reset_tm(self):
        """
        Reset the learning state.
        """
        self.tm.reset()
