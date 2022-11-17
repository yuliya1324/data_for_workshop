import pandas as pd
import re
from tqdm.auto import tqdm


def gaw(sentence, attention, agent_ind, verb_ind):
  
  weights = {i:{j: None for j in range(12)} for i in range(12)}

  for l in range(len(attention)): # 12
    for k in range(len(attention[l][0])): # 2
      for i in range(len(attention[l][0][k])):
        if i == verb_ind:
          for j in range(len(attention[l][0][k][i])):
            if j == agent_ind:
              weights[l][k] = {(sentence.split(' ')[0], sentence.split(' ')[1]): float(attention[l][0][k][i][j])}

  return weights
  
  
def get_max_weights(weights):
  
  a, v = list(weights[0][0][0])[0][0], list(weights[0][0][0])[0][1]
  out_weights = {i: {j: {(a, v): 0 for _ in range(12)} for j in range(12)} for i in range(12)}

  for weight_range in range(len(weights)):
    for i in range(12):
      for j in range(12):
        if out_weights[i][j][(a, v)] < weights[weight_range][i][j][(a, v)]:
          out_weights[i][j][(a, v)] = weights[weight_range][i][j][(a, v)]

  return out_weights
  
  
def get_mean_weights(weights):
  
  a, v = list(weights[0][0][0])[0][0], list(weights[0][0][0])[0][1]
  out_weights = {i: {j: {(a, v): 0 for _ in range(12)} for j in range(12)} for i in range(12)}

  for weight_range in range(len(weights)):
    for i in range(12):
      for j in range(12):
        out_weights[i][j][(a, v)] += weights[weight_range][i][j][(a, v)]
  
  for i in range(12):
    for j in range(12):
      out_weights[i][j][(a, v)] = out_weights[i][j][(a, v)]/len(weights)

  return out_weights
  
  
def get_all_attention_weights(sentence, attention, agent_inds, verb_inds):

  out = {'mean_bw_vtokens': None, 'max_bw_vtokens': None, 'st_bw_vtokens': None}

  if len(verb_inds) == 1 and len(agent_inds) == 1: 
    for tp in out.keys():
      weights = gaw(sentence, attention, agent_inds[0], verb_inds[0])
      out[tp] = {'mean_bw_atokens':weights, 'max_bw_atokens':weights, 
                 'st_bw_atokens': weights}
  else:
    if len(verb_inds) == 1:
      ag_weights = []

      for i in range(len(agent_inds)):
        ag_weights.append(gaw(sentence, attention, agent_inds[i], verb_inds[0]))

      mean_ag_weights = get_mean_weights(ag_weights)
      max_ag_weights = get_max_weights(ag_weights)

      ag_out = {'mean_bw_atokens':mean_ag_weights, 'max_bw_atokens':max_ag_weights, 
                'st_bw_atokens': ag_weights[0]}

      for tp in out.keys():
        out[tp] = ag_out

    elif len(agent_inds) == 1:
      vb_weights = []

      for i in range(len(verb_inds)):
        vb_weights.append(gaw(sentence, attention, agent_inds[0], verb_inds[i]))

      mean_vb_weights = get_mean_weights(vb_weights)
      max_vb_weights = get_max_weights(vb_weights)

      out['mean_bw_vtokens'] = {'mean_bw_atokens':mean_vb_weights, 'max_bw_atokens':mean_vb_weights, 
                                'st_bw_atokens': mean_vb_weights}
      out['max_bw_vtokens'] = {'mean_bw_atokens':max_vb_weights, 'max_bw_atokens':max_vb_weights, 
                               'st_bw_atokens': max_vb_weights}
      out['st_bw_vtokens'] = {'mean_bw_atokens':vb_weights[0], 'max_bw_atokens':vb_weights[0], 
                              'st_bw_atokens': vb_weights[0]}


    else:
      av_weights = [] 
      for i in range(len(verb_inds)): 
        verb_ag = []
        for j in range(len(agent_inds)):
          verb_ag.append(gaw(sentence, attention, agent_inds[j], verb_inds[i]))
        av_weights.append(verb_ag)
      counted_ag_weights = []
      for av in av_weights:
        mean_av_weights = get_mean_weights(av)
        max_av_weights = get_max_weights(av)
        first_av_weights = av[0]
        counted_ag_weights.append([mean_av_weights, max_av_weights, first_av_weights])
      out['mean_bw_vtokens'] = {'mean_bw_atokens': get_mean_weights([_[0] for _ in counted_ag_weights]),
                                'max_bw_atokens': get_mean_weights([_[1] for _ in counted_ag_weights]), 
                                'st_bw_atokens': get_mean_weights([_[2] for _ in counted_ag_weights])}
      out['max_bw_vtokens'] = {'mean_bw_atokens': get_max_weights([_[0] for _ in counted_ag_weights]),
                                'max_bw_atokens': get_max_weights([_[1] for _ in counted_ag_weights]), 
                                'st_bw_atokens': get_max_weights([_[2] for _ in counted_ag_weights])}
      out['st_bw_vtokens'] = {'mean_bw_atokens': counted_ag_weights[0][0],
                                'max_bw_atokens': counted_ag_weights[0][1], 
                                'st_bw_atokens': counted_ag_weights[0][2]}

  return out            
  
  
def find_indexes(tokenizer, model, sentence, w):
    sent = sentence.lower()
    words = re.findall('[а-яё\-]+|[a-z\-]+|[^а-яёa-z0-9\-]|[0-9\-]+', sent)
    indss = [0]
    word_indexes = {}
    for word in words:
        if word != ' ':
            inputs = tokenizer.encode_plus(word,  return_tensors='pt')
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            attention = model(input_ids, token_type_ids=token_type_ids)[-1]
            input_id_list = input_ids[0].tolist() 
            tokens = tokenizer.convert_ids_to_tokens(input_id_list)
            del tokens[0]
            del tokens[-1]
            for i in range(len(tokens)):
                indss.append(indss[-1]+1)
            if word in w:
                word_indexes[word] = indss[-len(tokens)::]
    return word_indexes
    

def get_matrix(df, tokenizer, model, tqdm):

    verb_lists = ['mean_bw_vtokens', 'max_bw_vtokens', 'st_bw_vtokens']
    role_lists = ['mean_bw_atokens', 'max_bw_atokens', 'st_bw_atokens']

    dicts = ['text', 'target', 'verb', 'layer', 'head', 'role']
    for d in dicts:
        globals()[d] = {vtoken:{token:[] for token in role_lists} for vtoken in verb_lists}

    indexes = df.index

    for ind in tqdm(indexes, total=len(indexes)):
        sentence = df.loc[ind, 'sentence']
        inputs = tokenizer.encode_plus(sentence,  return_tensors='pt')
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]

        input_id_list = input_ids[0].tolist() 
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)

        role_ind = df.loc[ind, 'idx_target'].split('-')
        verb_ind = df.loc[ind, 'idx_head'].split('-')

        
        role_token = sentence[int(role_ind[0]):int(role_ind[1])].lower()
        verb_token = sentence[int(verb_ind[0]):int(verb_ind[1])].lower()

        words_indexes = find_indexes(tokenizer, model, sentence, [role_token.lower(), verb_token.lower()])

        role_indexes = words_indexes[role_token]
        verb_indexes = words_indexes[verb_token]

        role_name = df.loc[ind, 'role']

        all_types_of_weights = get_all_attention_weights(' '.join([role_token, verb_token]), attention, role_indexes, verb_indexes)

        for v in verb_lists:
            for a in role_lists:
                for l in list(all_types_of_weights[v][a]):
                    head_weights = []
                    for h in list(all_types_of_weights[v][a][l]):
                        head_weights.append(list(all_types_of_weights[v][a][l][h].values())[0])
      
                    text[v][a].append(sentence)
                    target[v][a].append(list(all_types_of_weights[v][a][0][0].keys())[0][0])
                    verb[v][a].append(list(all_types_of_weights[v][a][0][0].keys())[0][1])
                    layer[v][a].append(l)
                    head[v][a].append(head_weights)
                    role[v][a].append(role_name)

    for v in verb_lists:
        for a in role_lists:
            for i in range(12):
                try:
                    globals()['h_'+str(i)][v][a] = [h[i] for h in head[v][a]]
                except:
                    try:
                        globals()['h_'+str(i)][v] = {a: [h[i] for h in head[v][a]]}
                    except:
                        globals()['h_'+str(i)] = {v: {a: [h[i] for h in head[v][a]]}}

    columns = ['text', 'target', 'role', 'verb', 'layer']
    columns.extend(['h_'+str(i) for i in range(12)])

    for v in verb_lists:
        for a in role_lists:
            dt = {col: globals()[col][v][a] for col in columns}
            globals()['v'+v.split('_')[0]+'_a'+a.split('_')[0]] = pd.DataFrame.from_dict(dt, orient='index')
            globals()['v'+v.split('_')[0]+'_a'+a.split('_')[0]] = globals()['v'+v.split('_')[0]+'_a'+a.split('_')[0]].transpose()

    dframes = [vmean_amean, vmean_amax, vmean_ast, 
               vmax_amean, vmax_amax, vmax_ast, 
               vst_amean, vst_amax, vst_ast]
               
    dframes_names = ['vmean_amean', 'vmean_amax', 'vmean_ast', 
                     'vmax_amean', 'vmax_amax', 'vmax_ast', 
                     'vst_amean', 'vst_amax', 'vst_ast']
                     
    output_dfs = {dframes_names[i]:dframes[i] for i in range(9)}

    return output_dfs


def get_mean_df(df, mean):
  data = {'h_'+str(j):[] for j in range(12)}
  data['layer'] = []

  for i in range(12):
    for j in range(12):
      layer_df = df[df['layer']==i]
      col_name = 'h_'+str(j)
      data[col_name].append(mean(layer_df[col_name]))
    data['layer'].append(i)
  return pd.DataFrame(data)