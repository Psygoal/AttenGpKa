# -*- coding: utf-8 -*-
"""
@author: LXY
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EdgeNetwork(layers.Layer):
    
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True
    
    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features
    
    def get_config(self):
        config = super().get_config()
        return config
    
class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):

        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = tf.dynamic_partition(atom_features,molecule_indicator,self.batch_size)
        
        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    
class TransformerEncoderReadout_mol(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=128, dense_dim=256, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        inputs_, molecule_indicator, picked_atoms = inputs
        picked_atoms_ind = tf.stack([tf.range(tf.shape(picked_atoms)[0],dtype='int64'),picked_atoms],axis=1)
        x = self.partition_padding([inputs_, molecule_indicator])
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        num = tf.reduce_sum(tf.cast(padding_mask,'float32'),axis=-1,keepdims=True)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
#         averaged = tf.reduce_sum(x,axis=1) / num
        
        return tf.gather_nd(proj_output,picked_atoms_ind)

    def get_config(self):
        config = super().get_config()
        return config

class TransformerEncoderReadout_solv(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=128, dense_dim=256, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
#         self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        num = tf.reduce_sum(tf.cast(padding_mask,'float32'),axis=-1,keepdims=True)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        averaged = tf.reduce_sum(proj_output,axis=1) / num
#         attention_output = self.attention(x, x, attention_mask=padding_mask)


        return averaged

    def get_config(self):
        config = super().get_config()
        return config

class SelfAttention(layers.Layer):
    def __init__(self, num_heads=8, embed_dim=128, dense_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="leaky_relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()
        
    def call(self, inputs):
        padding_mask = tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)
        num = tf.reduce_sum(tf.cast(padding_mask,'float32'),axis=-1,keepdims=True)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        averaged = tf.reduce_sum(proj_output,axis=1) / num
        return averaged
    
    def get_config(self):
        config = super().get_config()
        return config
    

class MessagePassing(layers.Layer):
    def __init__(self, units, steps=3, **kwargs): 
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length,activation='leaky_relu')
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])
        

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "steps": self.steps,
        })
        return config
    
def MPNNModel_mol(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=32,
    message_steps=3,
    mp_counts = 3,
    num_heads=8,
    dense_dim=256
):
    
    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    picked_atom_indices = layers.Input((), dtype="int64", name="picked_atom")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    
    atom_features_updated = layers.Dense(message_units*2,activation='leaky_relu')(atom_features)
    atom_features_updated = layers.Dense(message_units,activation='leaky_relu')(atom_features_updated)
    atom_features_updated = layers.Dense(message_units,activation='leaky_relu')(atom_features_updated)
    h = atom_features_updated
    
    for _ in range(mp_counts):
        
        h = MessagePassing(message_units, message_steps)(
            [h, bond_features, pair_indices]
        )

    h = TransformerEncoderReadout_mol(num_heads,message_units,dense_dim,batch_size)([h, molecule_indicator, picked_atom_indices])
    
    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, picked_atom_indices,molecule_indicator],
        outputs=[h],
    )

    return model

def MPNNModel_solv(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=32,
    message_steps=3,
    mp_counts = 3,
    num_heads=8,
    dense_dim=256
):
    
    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features_solv")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features_solv")
    pair_indices = layers.Input((2), dtype="int64", name="pair_indices_solv")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator_solv")
    
    atom_features_updated = layers.Dense(message_units*2,activation='leaky_relu')(atom_features)
    atom_features_updated = layers.Dense(message_units,activation='leaky_relu')(atom_features_updated)
    atom_features_updated = layers.Dense(message_units,activation='leaky_relu')(atom_features_updated)
    h = atom_features_updated
    
    for _ in range(mp_counts):
        
        h = MessagePassing(message_units, message_steps)(
            [h, bond_features, pair_indices]
        )

    h = TransformerEncoderReadout_solv(num_heads,message_units,dense_dim,batch_size)([h, molecule_indicator])
    
    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[h],
    )

    return model


def PredictionModel(atom_dim,bond_dim,batch_size,message_units=128,message_steps=3,mp_counts=3,num_heads=8,dense_dim=256,l2_param=0.1):
    
    mol_atom_features = layers.Input((atom_dim), dtype="float32",name='mol_atom_features')
    mol_bond_features = layers.Input((bond_dim), dtype="float32",name='mol_bond_features')
    mol_pair_indices = layers.Input((2), dtype="int32",name='mol_pair_indices')
    selected_atom_indices = layers.Input((), dtype="int64",name='mol_selected_atom_indices')
    mol_molecule_indicator = layers.Input((), dtype="int32",name='mol_mol_molecule_indicator')
    
    solv1_atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features_solv1")
    solv1_bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features_solv1")
    solv1_pair_indices = layers.Input((2), dtype="int64", name="pair_indices_solv1")
    solv1_molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator_solv1")
    
    solv2_atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features_solv2")
    solv2_bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features_solv2")
    solv2_pair_indices = layers.Input((2), dtype="int64", name="pair_indices_solv2")
    solv2_molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator_solv2")
    
    concentrations = layers.Input((3),dtype="float32", name="concentrations")
    
    # make prediction
    mpnnformol = MPNNModel_mol(atom_dim,bond_dim,batch_size,message_units,message_steps,mp_counts,num_heads,dense_dim)
    mpnnforsolv = MPNNModel_solv(atom_dim,bond_dim,batch_size,message_units,3,3,num_heads,dense_dim)
    
    mol_readout = mpnnformol((mol_atom_features,mol_bond_features,mol_pair_indices,selected_atom_indices,mol_molecule_indicator))
    solv1_readout = mpnnforsolv((solv1_atom_features, solv1_bond_features, solv1_pair_indices, solv1_molecule_indicator))
    solv2_readout = mpnnforsolv((solv2_atom_features, solv2_bond_features, solv2_pair_indices, solv2_molecule_indicator))
    
    concat_readout = layers.Concatenate(axis=1)([tf.expand_dims(mol_readout,axis=1),tf.expand_dims(solv1_readout,axis=1),tf.expand_dims(solv2_readout,axis=1)])
    concentrations_reshape = layers.Reshape([3,1])(concentrations)
    readout_weighted = layers.Multiply()([concat_readout,concentrations_reshape])
    attention_output = SelfAttention(num_heads=num_heads,embed_dim=message_units,dense_dim=dense_dim)(readout_weighted)
    readout_weighted_flatten = layers.Flatten()(attention_output)
    
    
    make_prediction = layers.Dense(100,activation='leaky_relu',kernel_regularizer=keras.regularizers.l2(l2_param))(readout_weighted_flatten)
    make_prediction = layers.Dense(1)(make_prediction)
    
    # build model
    model = keras.Model(
        inputs=[mol_atom_features, mol_bond_features, mol_pair_indices, selected_atom_indices, mol_molecule_indicator,\
             solv1_atom_features, solv1_bond_features, solv1_pair_indices, solv1_molecule_indicator,\
             solv2_atom_features, solv2_bond_features, solv2_pair_indices, solv2_molecule_indicator,concentrations],
        outputs=[make_prediction],
    )
    return model