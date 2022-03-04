"""Create contextualized word embeddings."""
from typing import List,Any

from torch import embedding
from sentence_transformers import SentenceTransformer

models = {'bert':'bert-base-uncased',
          'roberta':'sentence-transformers/stsb-roberta-large'}
class modelEmbeddings:
    def __init__(self, model_type:str, load_direct= False)-> None:
        """Inititalize model embeddings.
        
        Parameters
        ----------

        model_type: str
            Type of transformer model to use
            Options: bert, roberta, elmo
        """
        self.model_type = model_type
        self.load_direct = load_direct
    
    def load_transformer_model(self)->Any:
        """Load transformer model.
        
        Returns
        -------
        model: Any
            Transformer model
        """
        if self.load_direct:
            model = SentenceTransformer(self.model_type)
        else:
            model = SentenceTransformer(models[self.model_type])
        return model

    def __call__(self, documents: List[str]) -> Any:
        """Output contextualized word embeddings.

        Parameters
        ----------
        documents: List[str]
            text to create embeddings
        
        Returns
        -------
        embeddings: torch.Tensor
            Word embeddings
        """
        model = self.load_transformer_model()
        embeddings = model.encode(documents)
        return embeddings


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/sample.csv')
    model_embeddings = modelEmbeddings('bert')
    embeddings = model_embeddings(df['text'])
    print(embeddings)

