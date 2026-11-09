import unittest

from muse_attention_bench.timing import ColdL2, event_replay_us, replay_orders


class TimingTests(unittest.TestCase):
    def test_replay_orders_and_invalid_counts(self):
        self.assertEqual(replay_orders(4), ["AB", "BA", "AB", "BA"])
        for invalid in [0, 2, 3, 5, 4.0, True]:
            with self.assertRaises(ValueError):
                replay_orders(invalid)

    def test_flush_occurs_before_each_single_replay_outside_events(self):
        events = []

        class Event:
            def __init__(self, **kwargs): pass
            def record(self): events.append("event")
            def synchronize(self): events.append("sync")
            def elapsed_time(self, other): return .007

        class Cuda: pass
        class Torch: pass
        class Cold:
            def read(self): events.append("flush")
        class Graph:
            def replay(self): events.append("replay")
        Cuda.Event = Event
        Torch.cuda = Cuda
        self.assertEqual(event_replay_us(Torch, Graph(), Cold()), 7)
        self.assertEqual(events, ["flush", "event", "replay", "event", "sync"])

    def test_too_small_l2_buffer_rejected_without_torch(self):
        with self.assertRaises(ValueError):
            ColdL2(None, 1 << 20)


if __name__ == "__main__":
    unittest.main()
