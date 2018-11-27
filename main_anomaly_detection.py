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
    

    # arguments for Temporal Memory
    

    args = parser.parse_args()

    return args

def main():
    """
    Main function to run through program
    """
    args = create_parser()

    np.random_seed(args.random_seed)

if __name__ == '__main__':
    main()
