from asyncio import proactor_events
from cProfile import label
import matplotlib.pyplot as plt





def main():
    processes = [4, 8, 16, 32, 64]
    time_mpi = [1.006876, 1.358434, 1.522107, 1.679666, 2.041392]
    time_openmp = [0.658103, 0.217466, 0.134899, 0.149378, 0.104514]
    time_pthreads = [0.182199, 0.290315, 0.456031, 0.900783, 1.809716]
    time_seq = [0.283313] * len(processes)
    speedup_mpi = [(x/y) for x,y in zip(time_seq, time_mpi)]
    speedup_openmp = [(x/y) for x,y in zip(time_seq, time_openmp)]
    speedup_pthreads = [(x/y) for x,y in zip(time_seq, time_pthreads)]

    plt.plot(processes, time_seq, 'o--', label="Sequential")
    plt.plot(processes, time_mpi, 'o--', label="MPI")
    plt.plot(processes, time_openmp, 'o--', label="OpenMP")
    plt.plot(processes, time_pthreads, 'o--', label="Pthreads")
    plt.legend()
    plt.xticks(processes)
    plt.xlabel("Number of Processes/Threads")
    plt.ylabel("Time (seconds)")
    plt.title("Time taken for GE, Matrix Size = 1024 * 1024")
    plt.tight_layout()
    plt.savefig("./time-proj2.png")
    plt.cla()
    plt.clf()

    plt.plot(processes, speedup_mpi, 'o--', label="MPI")
    plt.plot(processes, speedup_openmp, 'o--', label="OpenMP")
    plt.plot(processes, speedup_pthreads, 'o--', label="Pthreads")
    plt.legend()
    plt.xticks(processes)
    plt.xlabel("Number of Processes/Threads")
    plt.ylabel("Speedup")
    plt.title("Speedup for GE, Matrix Size = 1024 * 1024")
    plt.tight_layout()
    plt.savefig("./speedup-proj2.png")
    plt.cla()
    plt.clf()


if __name__=="__main__":
    main()