# Federated Edge Learning Simulation Diagram

```mermaid
flowchart LR

    GTSRB["GTSRB Dataset<br/>Traffic Sign Frames"] --> Camera
    CIFAR["CIFAR-10 Dataset<br/>Image Frames"] --> Camera

    Camera["Camera / Video Stream"] --> Timing

    KDE["KDE Delay Generator<br/>Empirical Interarrival Delays<br/>Scott's Rule Bandwidth"] --> Timing

    Timing["Frame Arrival Timing"] --> D1
    Timing --> D2
    Timing --> D3

    subgraph Edge1["Edge Agent 1"]
        D1["Local Frame Subset"] --> NN1
        C1["Batch Size = 64<br/>Epochs = 1"] --> NN1
        NN1["SimpleCNN<br/>Conv 3→16<br/>Conv 16→32<br/>FC 128→43"] --> T1
        T1["Local Training<br/>time.sleep(sample_delay())"] --> U1
        U1["Upload Weights<br/>Peer-to-Peer Latency"]
    end

    subgraph Edge2["Edge Agent 2"]
        D2["Local Frame Subset"] --> NN2
        C2["Batch Size = 64<br/>Epochs = 1"] --> NN2
        NN2["SimpleCNN<br/>Conv 3→16<br/>Conv 16→32<br/>FC 128→43"] --> T2
        T2["Local Training<br/>time.sleep(sample_delay())"] --> U2
        U2["Upload Weights<br/>Peer-to-Peer Latency"]
    end

    subgraph Edge3["Edge Agent 3"]
        D3["Local Frame Subset"] --> NN3
        C3["Batch Size = 64<br/>Epochs = 1"] --> NN3
        NN3["SimpleCNN<br/>Conv 3→16<br/>Conv 16→32<br/>FC 128→43"] --> T3
        T3["Local Training<br/>time.sleep(sample_delay())"] --> U3
        U3["Upload Weights<br/>Peer-to-Peer Latency"]
    end

    U1 --> Agg
    U2 --> Agg
    U3 --> Agg

    Agg["Federated Averaging<br/>Average Agent Weights"] --> Global

    Global["Updated Global Model"] --> DL1
    Global --> DL2
    Global --> DL3

    DL1["Download Updated Model"] --> NN1
    DL2["Download Updated Model"] --> NN2
    DL3["Download Updated Model"] --> NN3

    Global --> Eval
    Eval["Evaluate on GTSRB Test Set"] --> Acc
    Eval --> RoundTime

    Acc["Round Accuracy"]
    RoundTime["Round Time"]
```
