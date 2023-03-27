from Q1_random_generate import build_plot
from Q2_pla import Run_PLA
from Q3_pocket_pla import Pocket_vs_pla
from Q4_mislabel import Pocket_with_mislabel
from functions import random_generate
import sys

if __name__ == '__main__':
    print("y = mx + b")
    print("Please enter the value of m")
    m = int(sys.stdin.readline())
    print("Please enter the value of b")
    b = int(sys.stdin.readline())
    
    # Q1
    print("\n--------------Q1--------------\n")
    build_plot(m,b,30)
    
    print("\n--------------Q2--------------\n")
    Run_PLA(m, b, num_sample=30, times=3)
    
    print("\n--------------Q3--------------\n")
    Pocket_vs_pla(m, b, num_sample=2000, threshold=10000, con_threshold=1000)
    
    print("\n--------------Q4--------------\n")
    Pocket_with_mislabel(m, b, num_sample=2000, incorrect_num=50 , threshold=10000, con_threshold=500)