# memory = the route of "once upon a day"; a new generation "once upon a" fits its next word to the memory's next route
import sys; sys.path.insert(0,"research"); import numpy as np, settle as S, routes as RT
from opengpt import data
S.RUN=S.ROOT/"runs"/"20261122-071357"; S.T,S.NH=32,4; p,w,stoi=S.load()
e=lambda s:[stoi[x] for x in data.tokenize(s)]
_,memory,_=RT.forward(p,e("once upon a day"))          # the memory: route at every word
def fit(r,m): return float((r==m).mean())               # share of the 1,536 units that match
prompt=e("once upon a"); out=[]
for step in range(8):
    pos=len(prompt)+len(out)                             # the word about to be written
    logits,_,_=RT.forward(p,prompt+out); lp=logits-logits.max(); lp=lp-np.log(np.exp(lp).sum())
    if pos<len(memory):                                  # the memory still has a route for this place
        scores=[]
        for c in range(len(w)):
            _,r,_=RT.forward(p,prompt+out+[c]); scores.append((fit(r[pos],memory[pos]),lp[c],c))
        scores.sort(reverse=True)
        if step==0:
            print("| candidate next word | route fit to memory | model probability |\n|---|---|---|")
            for f,l,c in scores[:5]+[s for s in scores if w[s[2]]=="time"]: print(f"| {w[c]} | {f:.1%} | {np.exp(l):.1%} |")
        out.append(scores[0][2])
    else:
        out.append(int(lp.argmax()))
print("\nPrompt 2: once upon a\nResponse:", " ".join(w[i] for i in out))
