from models import BiDAF
from models import QANet



def make_model(log, model_name='BiDAF', **kwargs):
    log.info('Going to load the model {}...'.format(model_name))
    if (model_name == 'BiDAF'):
        model = BiDAF(word_vectors=kwargs['word_vectors'],
                      hidden_size=kwargs['hidden_size'],
                      drop_prob=kwargs['drop_prob'])
        return model
    elif model_name == 'QANet':        
        model = QANet(kwargs['word_vectors'],kwargs['char_vectors'], hidden_size=kwargs['hidden_size'])
        return model
   

