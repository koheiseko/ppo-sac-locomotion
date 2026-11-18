# PPO e SAC para locomoção

Projeto de Deep Reinforcement Learning aplicado à locomoção, com implementações de **PPO** (Proximal Policy Optimization) e **SAC** (Soft Actor-Critic) em PyTorch. O objetivo é explorar como agentes aprendem a controlar movimentos em espaços de ação contínuos por meio da interação com ambientes simulados.

Os experimentos utilizam o MuJoCo por meio do Gymnasium, com configurações para Walker2d, Hopper, HalfCheetah e Humanoid. O projeto explora duas abordagens Actor-Critic: o PPO, que aprende a partir de trajetórias coletadas pela política atual, e o SAC, que reutiliza experiências armazenadas e incorpora entropia ao objetivo para incentivar a exploração.

Licença MIT — veja [LICENSE](LICENSE).
