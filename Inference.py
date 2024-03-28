from transformers import pipeline 
import os

# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pretrained_tokenizer = pipeline("text2text-generation", model="./results/original-tokenizer/results-normal/checkpoint-236000")
finetuned_tokenizer =  pipeline("text2text-generation", model="./results/tokenizer-finetuned/results-extended/checkpoint-196500")

print("Pretrained:")
print(pretrained_tokenizer("1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6  6. Be2 e5 7. Nb3 Be7 8. O-O O-O  9. Be3 Be6 10. Qd2 Nbd7 11. a4 Rc8 12. a5 Qc7 13. Rfd1  Nc5 14. Nxc5 dxc5 15. Nd5 Bxd5 16. exd5", max_new_tokens = 200))
print("Finetuned:")
print(finetuned_tokenizer("[PGN] 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6  6. Be2 e5 7. Nb3 Be7 8. O-O O-O  9. Be3 Be6 10. Qd2 Nbd7 11. a4 Rc8 12. a5 Qc7 13. Rfd1  Nc5 14. Nxc5 dxc5 15. Nd5 Bxd5 16. exd5 [BOARD] White R_a1 R_d1 K_g1 P_b2 P_c2 Q_d2 B_e2 P_f2 P_g2 P_h2 B_e3 P_a5 P_d5 Black p_c5 p_e5 p_a6 n_f6 p_b7 q_c7 b_e7 p_f7 p_g7 p_h7 r_c8 r_f8 k_g8 [ATTACKS] White B_e2$p_a6 B_e3$p_c5 Black n_f6$P_d5 q_c7$P_a5", max_new_tokens = 200))
print("Dataset:")
print("Now white has a full passed pawn, and black has little compensation for it. Tornbetween Rd8 to threaten the d5 pawn, Bd6 to hold rank and improve queenmobility, and c4 to prevent white playing c4 first.  I don't like c4 asI cannot easily defend it and it gives white the threat of Bb6.  Bd6 allowsme to follow with Qe7 and Nd7 will allow me to start pushing back on thekingside.  Rc-d8 lessens the defence on the c5 pawn, which cannot now havea pawn for support. Rf-d8 makes a later f5 pawn push less effective, andleaves the c8 rook with nowhere to go. In each of these cases white canjust play c4 and I have no threats.  Bd6 with the intent of Qe7 and Nd7seems the only option, with hopefully an f4 push to follow. With the knightand bishop defending c5 and e5 the rooks should be free to get some actionagain.")

print("Pretrained:")
print(pretrained_tokenizer("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 Bc5 5. Nxf7  Bxf2+ 6. Kf1  Qe7 7. Nxh8 d5 8. exd5 Nd4 9. Kxf2 Ne4+  10. Kg1 Qh4 11. g3 Nxg3 12. hxg3 Qxg3+ 13. Kf1 Bh3+ 14. Rxh3 Qxh3+ 15. Kf2 Qh2+  16. Ke3 Qf4+", max_new_tokens = 200))
print("Finetuned:")
print(finetuned_tokenizer("[PGN] 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 Bc5 5. Nxf7  Bxf2+ 6. Kf1  Qe7 7. Nxh8 d5 8. exd5 Nd4 9. Kxf2 Ne4+  10. Kg1 Qh4 11. g3 Nxg3 12. hxg3 Qxg3+ 13. Kf1 Bh3+ 14. Rxh3 Qxh3+ 15. Kf2 Qh2+  16. Ke3 Qf4+ [BOARD] White R_a1 N_b1 B_c1 Q_d1 P_a2 P_b2 P_c2 P_d2 K_e3 B_c4 P_d5 N_h8 Black n_d4 q_f4 p_e5 p_a7 p_b7 p_c7 p_g7 p_h7 r_a8 k_e8 [ATTACK] White K_e3$n_d4 K_e3$q_f4 Black n_d4$P_c2 q_f4$K_e3", max_new_tokens = 200))
print("Dataset:")
print("And with triple repeat game was decided as draw.")

print("Pretrained:")
print(pretrained_tokenizer("1. e4  e5 2. Nf3 f6 3. Bc4  Ne7 4. d4 d5  5. exd5 e4  6. Nfd2 Nxd5 7. Bxd5 Qxd5 8. Nb3  e3", max_new_tokens = 200))
print("Finetuned:")
print(finetuned_tokenizer("[PGN] 1. e4  e5 2. Nf3 f6 3. Bc4  Ne7 4. d4 d5  5. exd5 e4  6. Nfd2 Nxd5 7. Bxd5 Qxd5 8. Nb3  e3 [BOARD] White R_a1 N_b1 B_c1 Q_d1 K_e1 R_h1 P_a2 P_b2 P_c2 P_f2 P_g2 P_h2 N_b3 P_d4 Black p_e3 q_d5 p_f6 p_a7 p_b7 p_c7 p_g7 p_h7 r_a8 n_b8 b_c8 k_e8 b_f8 r_h8 [ATTACKS] White B_c1$p_e3 P_f2$p_e3 Black p_e3$P_f2 q_d5$P_g2 q_d5$N_b3 q_d5$P_d4", max_new_tokens = 200))
print("Dataset:")
print("A discovered attack on the g2 pawn, as well as an attack on f2.")

print("Pretrained:")
print(pretrained_tokenizer("1. e4  d6 2. f4 Nf6 3. Nc3 g6 4. Nf3 Bg7 5. d4 O-O 6. Bd3 Na6 7. O-O c5 8. Be3  Ng4  9. Qd2 Nxe3 10. Qxe3 cxd4  11. Nxd4 Qb6", max_new_tokens = 200))
print("Finetuned:")
print(finetuned_tokenizer("[PGN] 1. e4  d6 2. f4 Nf6 3. Nc3 g6 4. Nf3 Bg7 5. d4 O-O 6. Bd3 Na6 7. O-O c5 8. Be3  Ng4  9. Qd2 Nxe3 10. Qxe3 cxd4  11. Nxd4 Qb6 [BOARD] White R_a1 R_f1 K_g1 P_a2 P_b2 P_c2 P_g2 P_h2 N_c3 B_d3 Q_e3 N_d4 P_e4 P_f4 Black n_a6 q_b6 p_d6 p_g6 p_a7 p_b7 p_e7 p_f7 b_g7 p_h7 r_a8 b_c8 r_f8 k_g8 [ATTACKS] White B_d3$n_a6 Black q_b6$P_b2 q_b6$N_d4 b_g7$N_d4", max_new_tokens = 200))
print("Dataset:")
print("Threatens Bxd4 to win the Queen. At the same time threatensat b2.   Or, how you can own the center and still be in immediate trouble!!")