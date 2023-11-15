from rdflib import Graph, Namespace, Literal
SEMANTIC_PROJECTION_ONTOLOGY = 'Log_PhD/dynamic_workspace/semantic_projection_ontology.ttl'
def Query_Ontology(file):
    # Create a graph and parse your RDF data
    g = Graph()
    g.parse(file, format="turtle")  # Replace "your_rdf_data.ttl" with the actual file name or data

    # Define the namespace
    ns1 = Namespace("http://example.org/")

    # SPARQL query
    query = """
    PREFIX ns1: <http://example.org/>

    SELECT ?theObject ?relation
    WHERE {
      ns1:dysarthria_0 ns1:the_object ?theObject ;
                       ns1:relation ?relation .
    }
    """

    # Execute the query
    results = g.query(query)

    # Print the results
    for row in results:
        the_object = row["theObject"]
        relation = row["relation"]

    o = the_object.split('/')[-1]
    r = relation.split('/')[-1]

    # print("object: "+o+"\n"+"relation: "+r)
    return r,o


if __name__ == '__main__':

    print(Query_Ontology(SEMANTIC_PROJECTION_ONTOLOGY)[1])