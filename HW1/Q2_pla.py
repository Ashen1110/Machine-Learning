from functions import (random_generate, PLA, verification, Plot)
import numpy as np

def Run_PLA(m, b, num_sample, times):
    count_ite = 0
    for i in range(times):
        samples, label = random_generate(m,b,num_sample)
        w = np.random.rand(1,3)
        
        # plt sample 
        title="HW1-2 PLA with 30 2D data"
        fig2 = Plot(samples, label, title)
        fig2.plt_line(
            w = None,
            m=m,
            b=b,
            num_samples=num_sample,
            iteration=None,
            label = f"Benchmark",
            content=""
        )
        
        # run pla
        result_w , iteration = PLA(samples, label, w, num_sample)
        verification(samples, label, result_w, num_sample, iteration, show=True)
        count_ite += iteration
        
        # plt pla
        fig2.plt_line(
            w=result_w,
            m=None,
            b=None,
            num_samples=num_sample,
            iteration=iteration,
            label=f"PLA",
            content=f", iteration={iteration}"
        )
        fig2.save_show(
            itr_avg= count_ite/3,
            filename=f"HW1-2_{i+1}.png",
            show_avg=True
        )
     
    print("Q2 is done!")

if __name__ == '__main__':
    m = 1
    b = 2
    
    # execution
    print("\n--------------Q2--------------\n")
    Run_PLA(m, b, num_sample=30, times=3)