model_paths = {
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
    'stylegan_church': 'pretrained_models/stylegan2-church-config-f.pt',
    'stylegan_horse': 'pretrained_models/stylegan2-horse-config-f.pt',
    'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
    'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
    'shape_predictor':
    'pretrained_models/shape_predictor_68_face_landmarks.dat',
    'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}

project_basedir = '/mnt/lustre/yslan/Repo/Research/SIGA22/BaseModels/StyleSDF'

for k, v in model_paths.items():
    model_paths[k] = f'{project_basedir}/project/utils/misc/' + model_paths[k]

model_paths['ir_se50_hwc'] = '/home/yslan/datasets/model_ir_se50.pth'
