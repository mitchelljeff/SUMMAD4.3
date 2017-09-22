from flask import Flask, jsonify, abort, request
import tensorflow as tf


from jtr.jack.data_structures import convert2qasettings
from jtr.jack.readers import readers
from jtr.load.embeddings import load_embeddings
from jtr.preprocess.vocab import Vocab



app = Flask(__name__)


et2rels={}
rel2qf={}

with open("pt-r2q.format") as f:
    for line in f:
        fields=line.split("\t")
        rel=fields[0]
        e1type=rel[0:3]
        qformat=fields[1]
        if e1type not in et2rels:
            et2rels[e1type]=[]
        et2rels[e1type].append(rel)
        rel2qf[rel]=qformat


emb = load_embeddings("data/pt_fasttext/wiki.pt.vec", "fasttext")
vocab = Vocab(emb=emb, init_from_embeddings=True)


reader = readers["fastqa_reader"](vocab, {"beam_size": 1, 'batch_size': 1,
                                       "max_support_length": None})
reader.setup_from_file("pt_fastqa")

@app.route('/api/pt_fastqa', methods=['POST'])
def get_relations():
    if not request.json:
        abort(400)
    support=request.json["text"]
    entities=request.json["nel"]["entities"]
    qas=[]
    for e in entities:
        entity=e["entity"]
        e1type=entity["type"].lower()
        name=entity["currlangForm"]
        rels=et2rels[e1type]
        for rel in rels:
            qid=name+"\t"+rel
            question=rel2qf[rel].format(name)
            qas.append({"question":{"text":question,"id":qid},"answers":[{"text":"","span":[0,0]}]})
    instances=[{"questions":qas,"support":[{"text":support}]}]      
    data ={'meta':'TACKBP','instances':instances}
    qa = convert2qasettings(data)
    answers = reader.process_outputs(qa, 1, debug=True)
    results = {qa[i][0].id: a.text for i, a in enumerate(answers)}
    return jsonify(results)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
