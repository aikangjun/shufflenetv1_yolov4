from quantization import *
from custom.CustomLayers import ConvBlock, ResBlock

#
# def convert(fp_model, model):
#     for i, (fp16_layer, layer) in enumerate(zip(fp_model.layers, model.layers)):
#         if isinstance(fp16_layer, CSPdarknet53Tiny):
#             for j, (inner_fp16_layer, inner_layer) in enumerate(zip(fp16_layer.layers, layer.layers)):
#                 if isinstance(inner_fp16_layer, ConvBlock):
#                     layer_judge(model.layers[i].layers[j].conv2d,
#                                 fp_model.layers[i].layers[j].conv2d)
#
#                     if fp_model.layers[i].layers[j].normalize:
#                         norm_judge(model.layers[i].layers[j].batch_norm,
#                                    fp_model.layers[i].layers[j].batch_norm)
#
#                 if isinstance(inner_fp16_layer, ResBlock):
#                     # init_conv
#                     layer_judge(model.layers[i].layers[j].init_conv.conv2d,
#                                 fp_model.layers[i].layers[j].init_conv.conv2d)
#
#                     if fp_model.layers[i].layers[j].init_conv.normalize:
#                         norm_judge(model.layers[i].layers[j].init_conv.batch_norm,
#                                    fp_model.layers[i].layers[j].init_conv.batch_norm)
#
#                     # former_middle_conv
#                     layer_judge(model.layers[i].layers[j].former_middle_conv.grouped_conv2d,
#                                 fp_model.layers[i].layers[j].former_middle_conv.grouped_conv2d)
#
#                     if fp_model.layers[i].layers[j].former_middle_conv.normalize:
#                         norm_judge(model.layers[i].layers[j].former_middle_conv.batch_norm,
#                                    fp_model.layers[i].layers[j].former_middle_conv.batch_norm)
#
#                     # latter_middle_conv
#                     layer_judge(model.layers[i].layers[j].latter_middle_conv.conv2d,
#                                 fp_model.layers[i].layers[j].latter_middle_conv.conv2d)
#
#                     if fp_model.layers[i].layers[j].latter_middle_conv.normalize:
#                         norm_judge(model.layers[i].layers[j].latter_middle_conv.batch_norm,
#                                    fp_model.layers[i].layers[j].latter_middle_conv.batch_norm)
#
#                     # final_conv
#                     layer_judge(model.layers[i].layers[j].final_conv.conv2d,
#                                 fp_model.layers[i].layers[j].final_conv.conv2d)
#
#                     if fp_model.layers[i].layers[j].final_conv.normalize:
#                         norm_judge(model.layers[i].layers[j].final_conv.batch_norm,
#                                    fp_model.layers[i].layers[j].final_conv.batch_norm)
#
#         if isinstance(fp16_layer, layers.Dense):
#             layer_judge(model.layers[i], fp_model.layers[i])
#
