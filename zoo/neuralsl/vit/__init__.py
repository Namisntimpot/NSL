def DinoV2(vit_type:str, torchhub:bool = False):
    '''
    vit: vits, vitb, vitl, vitg.  
    return: vit module, embed dim
    '''
    assert vit_type in ['vits', 'vitb', 'vitl', 'vitg']
    if torchhub:
        import torch
        pretrained = torch.hub.load(
            'zoo/neuralsl/vit/facebookresearch_dinov2_main',
            'dinov2_{:}14'.format(vit_type),
            source='local',
            pretrained=False)
        return pretrained, pretrained.blocks[0].attn.qkv.in_features
    else:
        from .dinov2 import DINOv2 as create_dinov2
        pretrained = create_dinov2(vit_type)
        return pretrained, pretrained.embed_dim