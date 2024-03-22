from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-236000")
model_tokenizer = AutoModelForSeq2SeqLM.from_pretrained("./results-tokenizer/checkpoint-236000")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
tokenizer_finetuned = AutoTokenizer.from_pretrained("/cfs/home/u024219/Tese/CARLSy/flanT5-finetuned")

inputs = tokenizer(["1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6  6. Be2 e5 7. Nb3 Be7 8. O-O O-O  9. Be3 Be6 10. Qd2 Nbd7 11. a4 Rc8 12. a5 Qc7 13. Rfd1  Nc5 14. Nxc5 dxc5 15. Nd5 Bxd5 16. exd5","1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 Bc5 5. Nxf7  Bxf2+ 6. Kf1  Qe7 7. Nxh8 d5 8. exd5 Nd4 9. Kxf2 Ne4+  10. Kg1 Qh4 11. g3 Nxg3 12. hxg3 Qxg3+ 13. Kf1 Bh3+ 14. Rxh3 Qxh3+ 15. Kf2 Qh2+  16. Ke3 Qf4+","1. e4  e5 2. Nf3 f6 3. Bc4  Ne7 4. d4 d5  5. exd5 e4  6. Nfd2 Nxd5 7. Bxd5 Qxd5 8. Nb3  e3","1. e4  d6 2. f4 Nf6 3. Nc3 g6 4. Nf3 Bg7 5. d4 O-O 6. Bd3 Na6 7. O-O c5 8. Be3  Ng4  9. Qd2 Nxe3 10. Qxe3 cxd4  11. Nxd4 Qb6"], return_tensors="pt")
inputs_2 = tokenizer_finetuned(["1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6  6. Be2 e5 7. Nb3 Be7 8. O-O O-O  9. Be3 Be6 10. Qd2 Nbd7 11. a4 Rc8 12. a5 Qc7 13. Rfd1  Nc5 14. Nxc5 dxc5 15. Nd5 Bxd5 16. exd5","1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 Bc5 5. Nxf7  Bxf2+ 6. Kf1  Qe7 7. Nxh8 d5 8. exd5 Nd4 9. Kxf2 Ne4+  10. Kg1 Qh4 11. g3 Nxg3 12. hxg3 Qxg3+ 13. Kf1 Bh3+ 14. Rxh3 Qxh3+ 15. Kf2 Qh2+  16. Ke3 Qf4+","1. e4  e5 2. Nf3 f6 3. Bc4  Ne7 4. d4 d5  5. exd5 e4  6. Nfd2 Nxd5 7. Bxd5 Qxd5 8. Nb3  e3","1. e4  d6 2. f4 Nf6 3. Nc3 g6 4. Nf3 Bg7 5. d4 O-O 6. Bd3 Na6 7. O-O c5 8. Be3  Ng4  9. Qd2 Nxe3 10. Qxe3 cxd4  11. Nxd4 Qb6"], return_tensors="pt")

outputs = model.generate(**inputs)
print("Pretrained:")
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

outputs = model_tokenizer.generate(**inputs_2)
print("Pretrained:")
print(tokenizer_finetuned.batch_decode(outputs, skip_special_tokens=True))

print("Dataset")
print("Now white has a full passed pawn, and black has little compensation for it. Tornbetween Rd8 to threaten the d5 pawn, Bd6 to hold rank and improve queenmobility, and c4 to prevent white playing c4 first.  I don't like c4 asI cannot easily defend it and it gives white the threat of Bb6.  Bd6 allowsme to follow with Qe7 and Nd7 will allow me to start pushing back on thekingside.  Rc-d8 lessens the defence on the c5 pawn, which cannot now havea pawn for support. Rf-d8 makes a later f5 pawn push less effective, andleaves the c8 rook with nowhere to go. In each of these cases white canjust play c4 and I have no threats.  Bd6 with the intent of Qe7 and Nd7seems the only option, with hopefully an f4 push to follow. With the knightand bishop defending c5 and e5 the rooks should be free to get some actionagain.")
print("And with triple repeat game was decided as draw.")
print("A discovered attack on the g2 pawn, as well as an attack on f2.")
print("Threatens Bxd4 to win the Queen. At the same time threatensat b2.   Or, how you can own the center and still be in immediate trouble!!")


