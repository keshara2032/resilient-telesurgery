import numpy as np

[{'subject': 'S02', 'accuracy': 0.3635122119928878}, {'subject': 'S03', 'accuracy': 0.7659433452761938}, {'subject': 'S04', 'accuracy': 0.8805786941543865}, {'subject': 'S05', 'accuracy': 0.86801959868764}, {'subject': 'S06', 'accuracy': 0.9051102117372641}, {'subject': 'S07', 'accuracy': 0.8438381760460217}, {'subject': 'S08', 'accuracy': 0.9628430563700051}, {'subject': 'S09', 'accuracy': 0.9832699464430954}]



[{'subject': 'S02', 'accuracy': 0.42546724712132644}]


[{'subject': 'S02', 'accuracy': 0.4320476640961142}, {'subject': 'S03', 'accuracy': 0.6364740289734879}, {'subject': 'S04', 'accuracy': 0.5566451260485671}, {'subject': 'S05', 'accuracy': 0.7316233099342232}, {'subject': 'S06', 'accuracy': 0.4860967423124986}, {'subject': 'S07', 'accuracy': 0.4860682258701214}, {'subject': 'S08', 'accuracy': 0.6410973494004979}, {'subject': 'S09', 'accuracy': 0.5033635290877394}]


[{'subject': 'S02', 'accuracy': 0.40563718671817434}, {'subject': 'S03', 'accuracy': 0.834682795222559}, {'subject': 'S04', 'accuracy': 0.8231309019327177}, {'subject': 'S05', 'accuracy': 0.9105368299006302}, {'subject': 'S06', 'accuracy': 0.8490217942512044}, {'subject': 'S07', 'accuracy': 0.8881389824739631}, {'subject': 'S08', 'accuracy': 0.9715609867562846}, {'subject': 'S09', 'accuracy': 0.8786580901602647}]
0.8201709459269747


[{'subject': 'S02', 'accuracy': 0.02561925795100488}, {'subject': 'S03', 'accuracy': 0.6857284225534439}, {'subject': 'S04', 'accuracy': 0.40097055688559347}, {'subject': 'S05', 'accuracy': 0.5470776787432532}, {'subject': 'S06', 'accuracy': 0.8080182303774237}, {'subject': 'S07', 'accuracy': 0.2361322403224366}, {'subject': 'S08', 'accuracy': 0.49778270915647804}, {'subject': 'S09', 'accuracy': 0.5582263820911745}]



# layers 2
results = [{'subject': 'S02', 'accuracy': 0.5762350209197186},
{'subject': 'S03', 'accuracy': 0.5735129704207157},
{'subject': 'S04', 'accuracy': 0.699314311653761},
{'subject': 'S05', 'accuracy': 0.6430248205852966},
{'subject': 'S06', 'accuracy': 0.6368211190491191},
{'subject': 'S07', 'accuracy': 0.6175091035967848},
{'subject': 'S08', 'accuracy': 0.6215140511732963},
{'subject': 'S09', 'accuracy': 0.5442426359585311}]

acc = []
for x in results:
    acc.append(x['accuracy'])
    
print(np.average(acc))

{'average' : 0.6140217541696529}
