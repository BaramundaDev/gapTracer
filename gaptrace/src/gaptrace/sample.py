"""A tiny hand-made corpus (invented papers) used by the tests and by
`gaptrace demo`. It contains one gap of each of the four kinds."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Tuple

from .bundle import PassageStore, write_bundle
from .schema import Fact, Link, Note, Passage, make_id


def _l(target, type_, pid, sentence, verified=True):
    return Link(target=target, type=type_, evidence_passage_id=pid, evidence_sentence=sentence,
                verified=verified, confidence=0.9 if verified else 0.2)


def sample_corpus() -> Tuple[List[Note], List[Passage]]:
    def nid(t, name):
        return make_id(t, name)

    P = nid("problem", "Forgetting in sequential fine-tuning")
    M1, M2, M3 = (nid("method", n) for n in ("Replay buffer", "Weight consolidation", "Adapter isolation"))
    D1, D2 = nid("dataset", "Split CIFAR-100"), nid("dataset", "Permuted MNIST")
    L1 = nid("limitation", "Tested only on vision tasks")
    L2 = nid("limitation", "Needs task boundaries at training time")
    Q1 = nid("open_question", "Does replay help language models")
    R1, R2 = nid("result", "Replay accuracy paper A"), nid("result", "Replay accuracy paper B")
    A, B, C = "pap-a", "pap-b", "pap-c"

    passages = [
        Passage(id="a-1", paper_id=A, section="method", date=date(2023, 3, 1),
                text="We propose a replay buffer to reduce forgetting in sequential fine-tuning. "
                     "We evaluate the replay buffer on Split CIFAR-100. Replay reaches 71.2% accuracy on Split CIFAR-100."),
        Passage(id="a-2", paper_id=A, section="limitations", date=date(2023, 3, 1),
                text="Our study is limited: the method was tested only on vision tasks. "
                     "Whether replay helps language models is left for future work."),
        Passage(id="b-1", paper_id=B, section="method", date=date(2023, 9, 1),
                text="We introduce weight consolidation for forgetting in sequential fine-tuning. "
                     "Weight consolidation is evaluated on Split CIFAR-100 and on Permuted MNIST. "
                     "In our runs the replay buffer reaches 65.0% accuracy on Split CIFAR-100."),
        Passage(id="b-2", paper_id=B, section="limitations", date=date(2023, 9, 1),
                text="Like prior work, our experiments were tested only on vision tasks. "
                     "The approach needs task boundaries at training time."),
        Passage(id="c-1", paper_id=C, section="method", date=date(2024, 2, 1),
                text="Adapter isolation targets forgetting in sequential fine-tuning and removes the need "
                     "for task boundaries at training time. We test adapter isolation on Permuted MNIST."),
    ]
    notes = [
        Note(id=A, type="paper", name="Replay for continual learning", l0="Paper A (2023): replay buffer",
             l1="Proposes a replay buffer against forgetting; tested on Split CIFAR-100.", paper_date=date(2023, 3, 1),
             links=[_l(M1, "proposes", "a-1", "We propose a replay buffer to reduce forgetting in sequential fine-tuning."),
                    _l(R1, "reports", "a-1", "Replay reaches 71.2% accuracy on Split CIFAR-100."),
                    _l(L1, "states", "a-2", "Our study is limited: the method was tested only on vision tasks."),
                    _l(Q1, "states", "a-2", "Whether replay helps language models is left for future work.")]),
        Note(id=B, type="paper", name="Weight consolidation revisited", l0="Paper B (2023): weight consolidation",
             l1="Introduces weight consolidation; evaluated on Split CIFAR-100 and Permuted MNIST.",
             paper_date=date(2023, 9, 1),
             links=[_l(M2, "proposes", "b-1", "We introduce weight consolidation for forgetting in sequential fine-tuning."),
                    _l(R2, "reports", "b-1", "In our runs the replay buffer reaches 65.0% accuracy on Split CIFAR-100."),
                    _l(L1, "states", "b-2", "Like prior work, our experiments were tested only on vision tasks."),
                    _l(L2, "states", "b-2", "The approach needs task boundaries at training time.")]),
        Note(id=C, type="paper", name="Adapter isolation", l0="Paper C (2024): adapter isolation",
             l1="Adapter isolation removes the need for task boundaries; tested on Permuted MNIST.",
             paper_date=date(2024, 2, 1),
             links=[_l(M3, "proposes", "c-1", "We test adapter isolation on Permuted MNIST."),
                    _l(L2, "addresses", "c-1", "Adapter isolation targets forgetting in sequential fine-tuning and removes the need for task boundaries at training time.")]),
        Note(id=P, type="problem", name="Forgetting in sequential fine-tuning",
             l0="Models forget earlier tasks when fine-tuned on new ones", passage_ids=["a-1", "b-1", "c-1"]),
        Note(id=M1, type="method", name="Replay buffer", l0="Stores past examples and replays them", paper_id=A,
             paper_date=date(2023, 3, 1), passage_ids=["a-1"],
             links=[_l(P, "targets", "a-1", "We propose a replay buffer to reduce forgetting in sequential fine-tuning."),
                    _l(D1, "evaluated_on", "a-1", "We evaluate the replay buffer on Split CIFAR-100.")]),
        Note(id=M2, type="method", name="Weight consolidation", l0="Penalises changes to important weights",
             paper_id=B, paper_date=date(2023, 9, 1), passage_ids=["b-1"],
             links=[_l(P, "targets", "b-1", "We introduce weight consolidation for forgetting in sequential fine-tuning."),
                    _l(D1, "evaluated_on", "b-1", "Weight consolidation is evaluated on Split CIFAR-100 and on Permuted MNIST."),
                    _l(D2, "evaluated_on", "b-1", "Weight consolidation is evaluated on Split CIFAR-100 and on Permuted MNIST.")]),
        Note(id=M3, type="method", name="Adapter isolation", l0="Gives each task its own adapter", paper_id=C,
             paper_date=date(2024, 2, 1), passage_ids=["c-1"],
             links=[_l(P, "targets", "c-1", "Adapter isolation targets forgetting in sequential fine-tuning and removes the need for task boundaries at training time."),
                    _l(D2, "evaluated_on", "c-1", "We test adapter isolation on Permuted MNIST.")]),
        Note(id=D1, type="dataset", name="Split CIFAR-100", l0="CIFAR-100 split into sequential tasks"),
        Note(id=D2, type="dataset", name="Permuted MNIST", l0="MNIST with a pixel permutation per task"),
        Note(id=L1, type="limitation", name="Tested only on vision tasks",
             l0="Methods were tested only on vision tasks", paper_id=A, passage_ids=["a-2", "b-2"]),
        Note(id=L2, type="limitation", name="Needs task boundaries at training time",
             l0="The approach needs task boundaries at training time", paper_id=B, passage_ids=["b-2"]),
        Note(id=Q1, type="open_question", name="Does replay help language models",
             l0="Whether replay helps language models is open", paper_id=A, passage_ids=["a-2"]),
        Note(id=R1, type="result", name="Replay accuracy paper A", l0="Replay buffer: 71.2% on Split CIFAR-100",
             paper_id=A, paper_date=date(2023, 3, 1), passage_ids=["a-1"],
             fact=Fact(method="Replay buffer", dataset="Split CIFAR-100", metric="accuracy", value=71.2, unit="%")),
        Note(id=R2, type="result", name="Replay accuracy paper B", l0="Replay buffer: 65.0% on Split CIFAR-100",
             paper_id=B, paper_date=date(2023, 9, 1), passage_ids=["b-1"],
             fact=Fact(method="Replay buffer", dataset="Split CIFAR-100", metric="accuracy", value=65.0, unit="%")),
    ]
    return notes, passages


def write_sample(data: str | Path) -> Path:
    data = Path(data)
    notes, passages = sample_corpus()
    write_bundle(data / "bundle", notes)
    store = PassageStore(data / "passages.jsonl")
    store.add(passages)
    return data
