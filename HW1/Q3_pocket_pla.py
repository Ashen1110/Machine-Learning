from functions import (random_generate, PLA, verification, Pocket, Plot)
import numpy as np
from time import time

def Pocket_vs_pla(m, b, num_sample, threshold, con_threshold):
    samples, label = random_generate(m,b,num_sample)
    print("data generate\n")
    w = np.random.rand(1,3)
    title="HW1-3 Pockect v.s. PLA with 1000 2D data"
    fig3 = Plot(samples, label, title)
    fig3.plt_line(
        w = None,
        m=m,
        b=b,
        num_samples=num_sample,
        iteration=None,
        label = f"Benchmark",
        content=""
    )
    print("inital figure \n")
    #PLA
    PLA_start = time()
    PLA_w, PLA_iteration = PLA(samples, label, w, num_sample)
    PLA_exetime = time()-PLA_start
    PLA_error_rate = verification(samples, label, PLA_w, num_sample, iteration=PLA_iteration, show=False)
    # Plot PLA
    fig3.plt_line(
        w = PLA_w,
        m=None,
        b=None,
        num_samples=num_sample,
        iteration=PLA_iteration,
        label=f"PLA",
        content=f"\nerror rate = {PLA_error_rate:.03f}, iteration = {PLA_iteration}"
    )
    print("PLA done\n")
    # Pocket
    Pocket_start = time()
    Pocket_w, Pocket_iteration = Pocket(samples, label, w, num_sample, threshold, con_threshold)
    Pocket_exetime = time() - Pocket_start
    Pocket_error_rate = verification(samples, label, Pocket_w, num_sample, Pocket_iteration, show=False)
    #Plot pocket
    fig3.plt_line(
        w = Pocket_w,
        m=None,
        b=None,
        num_samples=num_sample,
        iteration=Pocket_iteration,
        label=f"Pocket",
        content=f"\nerror rate = {Pocket_error_rate:.03f}, iteration = {Pocket_iteration}"
    )
    print("Pocket done\n")
    
    #print screen
    print(f"PLA execution time = {PLA_exetime:.5f} sec")
    print(f"PLA iteration = {PLA_iteration}")
    print(f"PLA executin time per iteration = {PLA_exetime/PLA_iteration:.5f} sec")
    print(f"PLA error rate = {PLA_error_rate:.05f}\n")
    
    print(f"Pocket execution time = {Pocket_exetime:.5f} sec")
    print(f"Pocket iteration = {Pocket_iteration}")
    print(f"Pocket executin time per iteration = {Pocket_exetime/Pocket_iteration:.5f} sec")
    print(f"Pocket error rate = {Pocket_error_rate:.05f}\n")
    
    fig3.save_show(
            itr_avg= None,
            filename=f"HW1-3.png",
            show_avg=False
    )
     
    print("Q3 is done!")
    
if __name__ == '__main__':
    m = 1
    b = 2
    
    # execution
    print("\n--------------Q3--------------\n")
    Pocket_vs_pla(m, b, num_sample=2000, threshold=10000, con_threshold=1000)