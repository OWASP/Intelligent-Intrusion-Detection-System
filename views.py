from django.shortcuts import render

# Create your views here.
#import all packages from different files




class predict():
    def get(self):
                
        args = parser.parse_args()
                config_file = args['config']
        with open(config_file, "r") as f:
            config = json.load(f)
        
        input_data = args['data']
        model_name = config['model']
        model = model_name()

        if config['model_type'] = 'nn': 
            model = torch.load(filepath)
            _data = dataset.encode(input_data)
            _data = torch.from_numpy(np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)).reshape(-1, 1, 8, 8).cuda()
            _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
            response = dataset.decode(_out, label=True)    
            
        elif config['model_type'] = 'ml': 
            model = load_model(model_file)
            response = model.predict()
        else:
            response = 'Model Type not choosen'
            
        output = {'prediction': response}
        
        return output


