class debugPrint:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
# dprint = debugPrint(True)
dprint = debugPrint(False)




#* train
debug_vis = False # True

# flow_save = True
# flow_save_fre = 1e3


#* test
test_flow_save = False #True
test_input_save = True
bool_debug_test = False #True  8 car



#* all
# input_vis= False # True #False #True
# flow_vis_bool = False # True #False # True



def print_list(a_list):
    import inspect
    frame = inspect.currentframe().f_back
    name_a_list=None
    for name, value in frame.f_locals.items():
        if value is a_list:
            name_a_list = name
    print(' ')
    if name_a_list is not None:
        print(f'    {name_a_list}')
    for i, v in enumerate(a_list):
        if i<3 or i > len(a_list)-3:
            print(f"    {i}: {v}")
        elif i==3:
            print(f"    ...")