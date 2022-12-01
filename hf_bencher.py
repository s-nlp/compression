from hf_bench.benchmark import PyTorchBenchmark
from hf_bench.benchmark_args import PyTorchBenchmarkArguments

def main():
    args = PyTorchBenchmarkArguments(models=["distilbert-base-uncased"], batch_sizes=[1,16], 
                                sequence_lengths=[8, 32],
                                training=False, inference=True, cuda=True,
                                multi_process=True, verbose=False, trace_memory_line_by_line=False,
                                #inference_time_csv_file=training_args.output_dir+r'/inference_time.csv',
                                #inference_memory_csv_file=training_args.output_dir+r'/inference_memory.csv',
                                #train_time_csv_file=training_args.output_dir+r'/train_time.csv',
                                #train_memory_csv_file=training_args.output_dir+r'/train_memory.csv',
                                )
    benchmark = PyTorchBenchmark(args)
    benchmark.run()

if __name__ == "__main__":
    main()