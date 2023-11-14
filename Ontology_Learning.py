import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen import predict

CSV_SUBJECT = "Log_PhD/dynamic_workspace/ontology_dataset.csv"
Ontology_Learning_Model = 'Log_PhD/ontology/link_prediction.model'

def train(dataset):
    # read data from pykeen dataset method
    df = pd.read_csv(dataset)
    # print(df.sample(10))

    # Generate triples from the graph data
    tf = TriplesFactory.from_labeled_triples(df.values)

    # split triples into train and test
    training, testing = tf.split([0.8, 0.2], random_state=42)

    # generate embeddings using PyKEEN's pipeline method
    result = pipeline(
        training=training,
        testing=testing,
        model="TransR",
        model_kwargs=dict(embedding_dim=128),
        training_kwargs=dict(num_epochs=200),
        random_seed=42)

    # get entity labels from training set
    entity_labels = training.entity_labeling.all_labels()
    # convert entities to ids
    entity_ids = torch.as_tensor(training.entities_to_ids(entity_labels))
    # retrieve the embeddings using entity ids
    entity_embeddings = result.model.entity_representations[0](indices=entity_ids)
    # create a dictionary of entity labels and embeddings
    entity_embeddings_dict = dict(zip(entity_labels, entity_embeddings.detach().numpy()))

    # get relation labels from training set
    relation_labels = training.relation_labeling.all_labels()
    # convert relations to ids
    relation_ids = torch.as_tensor(training.relations_to_ids(relation_labels))
    # retrieve the embeddings using relation ids
    relation_embeddings = result.model.relation_representations[0](indices=relation_ids)
    # create a dictionary of relation labels and embeddings
    relation_embeddings_dict = dict(zip(relation_labels, relation_embeddings.detach().numpy()))
    return result


def predictRelation(myhead,mytail):
    result = train(CSV_SUBJECT)
    # tail prediction
    # head="the box"
    # tail = "three  sweaters"
    print(predict.predict_target(model=result.model,
                                 head=myhead,
                                 tail=mytail,
                                 triples_factory=result.training).df.head(7))



