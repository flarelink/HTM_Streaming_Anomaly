# =============================================================================
# main_anomaly_detection.py - Main program for anomaly detection using HTM
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

import HTM
import numpy as np
import argparse 

def create_parser():
    """
    Creates parser for command line inputs
    """
    parser = argparse.ArgumentParser(description='Parser for HTM streaming anomaly detection')

    # function to allow parsing of true/false boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')       

    """
    Hierarchical Temporal Memory arguments
    """

    # arguments for both Spatial Pooler and Temporal Memory
    parser.add_argument('--n_cols', type=int, default=2048,
                        help='Number of mini-columns initialized; default=2048')

    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed initialization for repeatability of tests; default=42')

    # arguments for Spatial Pooler
    parser.add_argument('--input_window', type=int, default=22,
                        help='Input window size; default=22')

    parser.add_argument('--pot_radius', type=int, default=16,
                        help='SP receptive field radius; default=16')

    parser.add_argument('--pot_percent', type=float, default=0.5,
                    help='Percentage of inputs that mini-columns are connected to; default=0.5')

    parser.add_argument('--inhibition', type=str2bool, nargs='?', default=True,
                        help='Specifies if global inhibition occurs; default=True')

    parser.add_argument('--act_cols_area', type=float, default=40,
                        help='Specifies number of active columns per inhibition area; default=40')

    parser.add_argument('--stim_thresh', type=int, default=0,
                        help='Specifies number of synapses needed for mini-column to become active; default=0')
    
    parser.add_argument('--perm_dec_SP', type=float, default=0.008,
                        help='Permanence decrement for SP; default=0.008')
    
    parser.add_argument('--perm_inc_SP', type=float, default=0.05,
                        help='Permanence increment for SP; default=0.05')
    
    parser.add_argument('--perm_thresh', type=float, default=0.1,
                        help='Permanence threshold for synapse to be formed; default=0.1')
    
    parser.add_argument('--boosting', type=float, default=0.0,
                        help='Boosting factor; default=0.0')

    # arguments for Temporal Memory 
    parser.add_argument('--n_cells_per_col', type=int, default=10,
                        help='Number of cells per mini-column; default=10')

    parser.add_argument('--init_perm', type=float, default=0.11,
                        help='Initial permanence values for newly created synapses; default=0.11')

    parser.add_argument('--min_thresh', type=int, default=8,
                        help='Minimum threshold of active synapses for a segment when searching for best-matching segments; default=8')

    parser.add_argument('--new_syn_learning', type=int, default=15,
                        help='Max number of synapses added to a segment during training; default=15')

    parser.add_argument('--perm_inc_TM', type=float, default=0.10,
                        help='Permanence increment for TM; default=0.10')

    parser.add_argument('--perm_dec_TM', type=float, default=0.10,
                        help='Permanence decrement for TM; default=0.10')

    parser.add_argument('--age', type=int, default=100000,
                        help='Number of iterations before global decay takes effect; default=100000')

    parser.add_argument('--decay', type=float, default=0.10,
                        help='Decay factor on permanences; default=0.10')

    parser.add_argument('--act_thresh', type=int, default=12,
                        help='Number of synapses that must be active to activate a segment; default=12')

    parser.add_argument('--pam_l', type=int, default=1,
                        help='Number of time steps to remain in "Pay Attention Mode"; default=1')

    parser.add_argument('--max_seg_per_cell', type=int, default=-1,
                        help='Maximum number of segments allowed per cell; default=-1')

    parser.add_argument('--max_syn_per_seg', type=int, default=-1,
                        help='Maximum number of synapses per segment; default=-1')

    args = parser.parse_args()

    return args

def main():
    """
    Main function to run through program
    """
    
    # create args
    args = create_parser()

    # initialize HTM model
    model = HTM.init_HTM(

            # SP inputs
            args.input_window,
            args.n_cols,
            args.pot_radius,
            args.pot_percent,
            args.inhibition,
            args.act_cols_area,
            args.stim_thresh,
            args.perm_dec_SP,
            args.perm_inc_SP,
            args.perm_thresh,
            args.boosting,
            args.random_seed,

            # TM inputs
            args.n_cells_per_col,
            args.init_perm,
            args.min_thresh,
            args.new_syn_learning,
            args.perm_inc_TM,
            args.perm_dec_TM,
            args.age,
            args.decay,
            args.act_thresh,
            args.pam_l,
            args.max_seg_per_cell,
            args.max_syn_per_seg
            )

    print(model)

if __name__ == '__main__':
    main()
