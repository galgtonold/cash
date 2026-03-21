"""Batch 62: Concurrency & threading — cash caching with threads, locks, queues."""
import textwrap
import pytest


@pytest.mark.stress
class TestThreadingBasics:
    """Test threading patterns across cells."""

    def test_thread_pool_results(self, nb_runner):
        """ThreadPoolExecutor results cached across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor

                def compute_square(x):
                    return x ** 2

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(compute_square, i) for i in range(10)]
                    results = [f.result() for f in futures]
                print(f"results={results}")
            """),
            textwrap.dedent("""\
                total = sum(results)
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]" in nb_runner.get_output(1)
        assert "total=285" in nb_runner.get_output(2)

    def test_thread_pool_map(self, nb_runner):
        """ThreadPoolExecutor.map across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor

                def process(item):
                    return item.upper()

                words = ['hello', 'world', 'from', 'threads']
                with ThreadPoolExecutor(max_workers=2) as pool:
                    processed = list(pool.map(process, words))
                print(f"processed={processed}")
            """),
            textwrap.dedent("""\
                joined = ' '.join(processed)
                print(f"joined={joined}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "processed=['HELLO', 'WORLD', 'FROM', 'THREADS']" in nb_runner.get_output(1)
        assert "joined=HELLO WORLD FROM THREADS" in nb_runner.get_output(2)


@pytest.mark.stress
class TestQueuePatterns:
    """Test queue-based patterns."""

    def test_queue_producer_consumer(self, nb_runner):
        """Queue-based producer/consumer pattern."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from queue import Queue
                import threading

                results = []
                q = Queue()

                def producer(items):
                    for item in items:
                        q.put(item)
                    q.put(None)  # sentinel

                def consumer():
                    while True:
                        item = q.get()
                        if item is None:
                            break
                        results.append(item * 2)

                t1 = threading.Thread(target=producer, args=([1, 2, 3, 4, 5],))
                t2 = threading.Thread(target=consumer)
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                print(f"results={sorted(results)}")
            """),
            textwrap.dedent("""\
                total = sum(results)
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[2, 4, 6, 8, 10]" in nb_runner.get_output(1)
        assert "total=30" in nb_runner.get_output(2)

    def test_thread_safe_counter(self, nb_runner):
        """Thread-safe counter with lock."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import threading

                class SafeCounter:
                    def __init__(self):
                        self._count = 0
                        self._lock = threading.Lock()
                    def increment(self, n=1):
                        with self._lock:
                            self._count += n
                    @property
                    def value(self):
                        return self._count

                counter = SafeCounter()
                threads = []
                for _ in range(10):
                    t = threading.Thread(target=lambda: [counter.increment() for _ in range(100)])
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
                print(f"count={counter.value}")
            """),
            textwrap.dedent("""\
                is_correct = counter.value == 1000
                print(f"correct={is_correct}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=1000" in nb_runner.get_output(1)
        assert "correct=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestProcessPoolPatterns:
    """Test process pool patterns (simulated with threading)."""

    def test_map_reduce_pattern(self, nb_runner):
        """Map-reduce pattern with thread pool."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor
                from collections import Counter

                documents = [
                    "the quick brown fox",
                    "the lazy dog",
                    "the quick dog jumps",
                    "brown fox jumps high"
                ]

                def map_words(doc):
                    return Counter(doc.split())

                with ThreadPoolExecutor() as pool:
                    word_counts = list(pool.map(map_words, documents))

                # Reduce
                total_counts = Counter()
                for wc in word_counts:
                    total_counts.update(wc)
            """),
            textwrap.dedent("""\
                top5 = total_counts.most_common(5)
                print(f"top5={top5}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "the" in out
        assert "3" in out  # "the" appears 3 times

    def test_parallel_aggregation(self, nb_runner):
        """Parallel data aggregation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor

                def chunk_sum(chunk):
                    return sum(chunk)

                data = list(range(1000))
                chunk_size = 100
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

                with ThreadPoolExecutor() as pool:
                    partial_sums = list(pool.map(chunk_sum, chunks))
                print(f"chunks={len(partial_sums)}")
            """),
            textwrap.dedent("""\
                total = sum(partial_sums)
                expected = sum(range(1000))
                print(f"total={total} expected={expected} match={total == expected}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chunks=10" in nb_runner.get_output(1)
        assert "match=True" in nb_runner.get_output(2)
