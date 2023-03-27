from functions import (random_generate, verification, Pocket, Plot)
import numpy as np
from time import time 

def Pocket_with_mislabel(m, b, num_sample, incorrect_num, threshold, con_threshold):
    samples, label = random_generate(m,b,num_sample)
    print("data generate\n")
    
    label = np.array(label)
    label[np.random.choice(np.arange(0, int(num_sample/2)), incorrect_num, replace=False)] = -1
    label[np.random.choice(np.arange(0, int(num_sample/2)), incorrect_num, replace=False)] = 1
    
    w = np.random.rand(1,3)
    title="HW1-4 Pockect with mislabel data"
    fig4 = Plot(samples, label, title)
    fig4.plt_line(
        w = None,
        m=m,
        b=b,
        num_samples=num_sample,
        iteration=None,
        label = f"Benchmark",
        content=""
    )
    print("inital figure \n")
    
    # Pocket
    Pocket_start = time()
    Pocket_w, Pocket_iteration = Pocket(samples, label, w, num_sample, threshold, con_threshold)
    Pocket_exetime = time() - Pocket_start
    Pocket_error_rate = verification(samples, label, Pocket_w, num_sample, Pocket_iteration, show=False)
    fig4.plt_line(
        w = Pocket_w,
        m=None,
        b=None,
        num_samples=num_sample,
        iteration=Pocket_iteration,
        label=f"Pocket",
        content=f"\nerror rate = {Pocket_error_rate:.03f}, iteration = {Pocket_iteration}"
    )
    print(f"Pocket execution time = {Pocket_exetime:.5f} sec")
    print(f"Pocket iteration = {Pocket_iteration}")
    print(f"Pocket executin time per iteration = {Pocket_exetime/Pocket_iteration:.5f} sec")
    print(f"Pocket error rate = {Pocket_error_rate:.05f}\n")
    
    fig4.save_show(
        itr_avg= None,
        filename=f"HW1-4.png",
        show_avg=False
    )
        
    print("Q4 is done!")
    
if __name__ == '__main__':
    m = 1
    b = 2
    
    # execution
    print("\n--------------Q4--------------\n")
    Pocket_with_mislabel(m, b, num_sample=2000, incorrect_num=50 , threshold=10000, con_threshold=500)