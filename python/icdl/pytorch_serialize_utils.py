
def pytorch_tensor_name_to_icdl_tensor_name(name):
    '''
        @param name: name string from model.state_dict().keys()
    '''
    parts = name.split(".")
    s = ""
    for i in range(len(parts) - 1):
        s += parts[i]
        s += "->"
    s = s[:-2] + "." + parts[-1]
    return s

