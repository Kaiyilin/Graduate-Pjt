import tensorflow as tf

prjt_configs = {
    "records" : {
        "log_path" : "./logs/",
        "ckpt_path" : "./ckpt_path/"
    },

    "data_path": {    
        "raw" : "./data_raw", 
        "normalised" : "./data_normalized"
        },

    "train": {
        "gen_filter_nums" : 40,
        "disc_filter_nums" : 20,
        "kernel_initialiser_dis" : tf.random_normal_initializer(0., 0.02),
        "kernel_size" : (3,3,3),
        "_lambda" : 100,
        "_batchSize" : 1,
        "_epochs" : 300,
        "d_lr" : 2e-6,
        "g_lr" : 2e-5
    },

    "model" : {
        "gen_input_shape" : (128, 128, 128, 1),
        "disc_input_shape" : (128, 128, 128, 2)
    }
}