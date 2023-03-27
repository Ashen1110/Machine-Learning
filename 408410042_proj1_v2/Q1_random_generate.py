from functions import random_generate
from functions import Plot

def build_plot(m,b,num_sample):
    samples, label = random_generate(m,b,num_sample)
    #x = samples[:,1]
    #y = samples[:,2]
    title="HW1-1 randomly generate 30 2D data samples"
    fig1 = Plot(samples, label, title)
    
    fig1.plt_line(
        w = None,
        m=m,
        b=b,
        num_samples=num_sample,
        iteration=None,
        label = f"Benchmark",
        content=""
    )
    
    fig1.save_show(
        itr_avg = None,
        filename= f"HW1-1",
        show_avg=False
    )
    print("Q1 is done!")


if __name__ == '__main__':
    m = 1
    b = 2
    
    # execution
    print("\n--------------Q1--------------\n")
    build_plot(m,b,30)




