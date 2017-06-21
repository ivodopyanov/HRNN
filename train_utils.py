# -*- coding: utf-8 -*-
import sys
import numpy as np
import keras.backend as K
from math import e, pow, isnan

ERROR_LIMIT = pow(e,5)

def copy_weights_encoder_to_predictor_wordbased(objects):
    encoder = objects['encoder']
    predictor = objects['predictor']
    predictor.get_layer('emb').embeddings.set_value(K.get_value(encoder.get_layer('emb').embeddings))
    copy_weights_encoder_to_predictor_charbased(objects)

def copy_weights_encoder_to_predictor_charbased(objects):
    encoder = objects['encoder']
    predictor = objects['predictor']
    predictor.get_layer('encoder').W_emb.set_value(K.get_value(encoder.get_layer('encoder').W_emb))
    predictor.get_layer('encoder').b_emb.set_value(K.get_value(encoder.get_layer('encoder').b_emb))
    predictor.get_layer('encoder').W.set_value(K.get_value(encoder.get_layer('encoder').W))
    predictor.get_layer('encoder').U.set_value(K.get_value(encoder.get_layer('encoder').U))
    predictor.get_layer('encoder').b.set_value(K.get_value(encoder.get_layer('encoder').b))
    predictor.get_layer('encoder').W1.set_value(K.get_value(encoder.get_layer('encoder').W1))
    predictor.get_layer('encoder').b1.set_value(K.get_value(encoder.get_layer('encoder').b1))
    predictor.get_layer('dense_0').kernel.set_value(K.get_value(encoder.get_layer('dense_0').kernel))
    predictor.get_layer('dense_0').bias.set_value(K.get_value(encoder.get_layer('dense_0').bias))
    predictor.get_layer('output').kernel.set_value(K.get_value(encoder.get_layer('output').kernel))
    predictor.get_layer('output').bias.set_value(K.get_value(encoder.get_layer('output').bias))


def copy_weights_rl_to_predictor(objects):
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    predictor.get_layer('encoder').W_action_1.set_value(K.get_value(rl_model.get_layer('encoder').W_action_1))
    predictor.get_layer('encoder').U_action_1.set_value(K.get_value(rl_model.get_layer('encoder').U_action_1))
    predictor.get_layer('encoder').b_action_1.set_value(K.get_value(rl_model.get_layer('encoder').b_action_1))
    predictor.get_layer('encoder').W_action_3.set_value(K.get_value(rl_model.get_layer('encoder').W_action_3))
    predictor.get_layer('encoder').b_action_3.set_value(K.get_value(rl_model.get_layer('encoder').b_action_3))

def copy_weights_rl_to_encoder(objects):
    encoder = objects['encoder']
    rl_model = objects['rl_model']
    encoder.get_layer('encoder').W_action_1.set_value(K.get_value(rl_model.get_layer('encoder').W_action_1))
    encoder.get_layer('encoder').U_action_1.set_value(K.get_value(rl_model.get_layer('encoder').U_action_1))
    encoder.get_layer('encoder').b_action_1.set_value(K.get_value(rl_model.get_layer('encoder').b_action_1))
    encoder.get_layer('encoder').W_action_3.set_value(K.get_value(rl_model.get_layer('encoder').W_action_3))
    encoder.get_layer('encoder').b_action_3.set_value(K.get_value(rl_model.get_layer('encoder').b_action_3))

def copy_weights_predictor_to_encoder(objects):
    encoder = objects['encoder']
    predictor = objects['predictor']
    encoder.get_layer('encoder').W_action_1.set_value(K.get_value(predictor.get_layer('encoder').W_action_1))
    encoder.get_layer('encoder').U_action_1.set_value(K.get_value(predictor.get_layer('encoder').U_action_1))
    encoder.get_layer('encoder').b_action_1.set_value(K.get_value(predictor.get_layer('encoder').b_action_1))
    encoder.get_layer('encoder').W_action_3.set_value(K.get_value(predictor.get_layer('encoder').W_action_3))
    encoder.get_layer('encoder').b_action_3.set_value(K.get_value(predictor.get_layer('encoder').b_action_3))


def copy_weights_predictor_evo_to_encoder(objects):
    encoder = objects['encoder']
    predictor = objects['predictor_evo']
    encoder.get_layer('encoder').W_action_1.set_value(K.get_value(predictor.get_layer('encoder').W_action_1))
    encoder.get_layer('encoder').U_action_1.set_value(K.get_value(predictor.get_layer('encoder').U_action_1))
    encoder.get_layer('encoder').b_action_1.set_value(K.get_value(predictor.get_layer('encoder').b_action_1))
    #encoder.get_layer('encoder').W_action_2.set_value(K.get_value(predictor.get_layer('encoder').W_action_2))
    #encoder.get_layer('encoder').b_action_2.set_value(K.get_value(predictor.get_layer('encoder').b_action_2))
    encoder.get_layer('encoder').W_action_3.set_value(K.get_value(predictor.get_layer('encoder').W_action_3))
    encoder.get_layer('encoder').b_action_3.set_value(K.get_value(predictor.get_layer('encoder').b_action_3))


def run_training2(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    predictor._make_predict_function()
    epoch_size = int(len(objects['train_indexes'])/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))

    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))

    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss1_total = []
        acc_total = []
        loss2_total = []
        depth_total = []
        lla_total = []
        #objects['encoder'].save_weights("encoder.h5")
        #objects['predictor'].save_weights("predictor.h5")
        #objects['rl_model'].save_weights("rl_model.h5")
        for j in range(epoch_size):

            '''if j == 605:
                objects['encoder'].save_weights("encoder.h5")
                objects['predictor'].save_weights("predictor.h5")
                objects['rl_model'].save_weights("rl_model.h5")
                return'''
            batch = next(objects['data_gen'])
            loss1 = encoder.train_on_batch(batch[0], batch[1])
            if isnan(loss1[0]):
                continue
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])

            if len(loss1_total) == 0:
                avg_loss1 = 0
            else:
                avg_loss1 = np.sum(loss1_total)*1.0/len(loss1_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)*1.0/len(acc_total)

            settings['copy_etp'](objects)

            ins = batch[0] + [1.]
            y_pred = predictor.predict_function(ins)

            output = y_pred[0]
            input_x = y_pred[1]
            input_h = y_pred[2]
            policy = y_pred[3]
            policy_calculated = y_pred[4]
            chosen_action = y_pred[5]
            policy_depth = y_pred[6]
            depth = y_pred[7]

            depth_total.append(depth[0])

            if np.sum(policy_calculated) > 0:
                error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), ERROR_LIMIT)
                #error = -np.log(np.sum(output*batch[1], axis=1))
                X,Y,sample_weight = restore_exp3(settings, input_x, error, input_h, policy, policy_calculated, chosen_action, policy_depth)
                loss2 = rl_model.train_on_batch(X,Y)
                if isnan(loss2):
                    continue
                loss2_total.append(loss2)
                copy_weights_rl_to_predictor(objects)
                copy_weights_rl_to_encoder(objects)


            if len(loss2_total) == 0:
                avg_loss2 = 0
            else:
                avg_loss2 = np.sum(loss2_total)*1.0/len(loss2_total)
            if len(depth_total) == 0:
                avg_depth = 0
            else:
                avg_depth = np.sum(depth_total)*1.0/len(depth_total)


            sys.stdout.write("\r batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, depth = {:.4f}"
                         .format(j+1, epoch_size,
                                 avg_loss1, avg_acc, avg_loss2, avg_depth))


        sys.stdout.write("\n")
        loss1_total = []
        acc_total = []
        loss2_total = []
        depth_total = []
        for i in range(val_epoch_size):
            batch = next(objects['val_gen'])
            try:
                loss1 = encoder.evaluate(batch[0], batch[1], batch_size=settings['batch_size'], verbose=0)
                y_pred = predictor.predict_on_batch(batch[0])
            except ValueError:
                sys.stdout.write("ValueError3!\n")
                continue
            if isnan(loss1[0]):
                continue

            output = y_pred[0]
            input_x = y_pred[1]
            input_h = y_pred[2]
            policy = y_pred[3]
            policy_calculated = y_pred[4]
            chosen_action = y_pred[5]
            policy_depth = y_pred[6]
            depth = y_pred[7]
            if np.sum(policy_calculated) > 0:
                error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), ERROR_LIMIT)
                #error = -np.log(np.sum(output*batch[1], axis=1))
                X,Y,sample_weight = restore_exp3(settings, input_x, error, input_h, policy, policy_calculated, chosen_action, policy_depth)
                loss2 = rl_model.evaluate(X,Y, batch_size=settings['batch_size'], verbose=0)
                if isnan(loss2):
                    continue
                loss2_total.append(loss2)
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])
            depth_total.append(depth[0])

            if len(loss1_total) == 0:
                avg_loss1 = 0
            else:
                avg_loss1 = np.sum(loss1_total)*1.0/len(loss1_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)*1.0/len(acc_total)
            if len(loss2_total) == 0:
                avg_loss2 = 0
            else:
                avg_loss2 = np.sum(loss2_total)*1.0/len(loss2_total)
            if len(depth_total) == 0:
                avg_depth = 0
            else:
                avg_depth = np.sum(depth_total)*1.0/len(depth_total)


            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, depth = {:.4f}"
                             .format(i+1, val_epoch_size,
                                     avg_loss1,
                                     avg_acc,
                                     avg_loss2,
                                     avg_depth))




def run_training_encoder_only(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))

    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))

    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss1_total = []
        acc_total = []
        for j in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = encoder.train_on_batch(batch[0], batch[1])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])

            if len(loss1_total) == 0:
                avg_loss1 = 0
            else:
                avg_loss1 = np.sum(loss1_total)*1.0/len(loss1_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)*1.0/len(acc_total)

            sys.stdout.write("\r batch {} / {}: loss1 = {:.4f}, acc = {:.4f}"
                         .format(j+1, epoch_size, avg_loss1, avg_acc))
        sys.stdout.write("\n")
        settings['copy_etp'](objects)
        loss1_total = []
        acc_total = []
        for i in range(val_epoch_size):
            batch = next(objects['val_gen'])
            loss1 = encoder.evaluate(batch[0], batch[1], batch_size=settings['batch_size'], verbose=0)

            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])
            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}"
                             .format(i+1, val_epoch_size,
                                     np.sum(loss1_total)*1.0/len(loss1_total),
                                     np.sum(acc_total)*1.0/len(acc_total)))



def run_training_RL_only(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    predictor._make_predict_function()
    epoch_size = int(len(objects['train_indexes'])/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))
    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))
    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss2_total = []
        depth_total = []
        for j in range(epoch_size):
            batch = next(objects['data_gen'])
            ins = batch[0] + [1.]
            y_pred = predictor.predict_function(ins)
            output = y_pred[0]
            input_x = y_pred[1]
            input_h = y_pred[2]
            policy = y_pred[3]
            policy_calculated = y_pred[4]
            chosen_action = y_pred[5]
            policy_depth = y_pred[6]
            depth = y_pred[7]

            if np.sum(policy_calculated) > 0:
                error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), ERROR_LIMIT)
                #error = -np.log(np.sum(output*batch[1], axis=1))
                X,Y,sample_weight = restore_exp3(settings, input_x, error, input_h, policy, policy_calculated, chosen_action, policy_depth)
                loss2 = rl_model.train_on_batch(X,Y)
                loss2_total.append(loss2)
                copy_weights_rl_to_predictor(objects)
                copy_weights_rl_to_encoder(objects)


            depth_total.append(depth[0])
            if len(loss2_total) == 0:
                avg_loss2 = 0
            else:
                avg_loss2 = np.sum(loss2_total)*1.0/len(loss2_total)
            if len(depth_total) == 0:
                avg_depth = 0
            else:
                avg_depth = np.sum(depth_total)*1.0/len(depth_total)
            sys.stdout.write("\r batch {} / {}: loss2 = {:.4f}, avg depth = {:.2f}".format(j+1, epoch_size, avg_loss2, avg_depth))

        sys.stdout.write("\n")
        loss1_total = []
        acc_total = []
        loss2_total = []
        depth_total = []
        for i in range(val_epoch_size):
            batch = next(objects['val_gen'])
            loss1 = encoder.evaluate(batch[0], batch[1], batch_size=settings['batch_size'], verbose=0)
            y_pred = predictor.predict_on_batch(batch[0])
            output = y_pred[0]
            input_x = y_pred[1]
            input_h = y_pred[2]
            policy = y_pred[3]
            policy_calculated = y_pred[4]
            chosen_action = y_pred[5]
            policy_depth = y_pred[6]
            depth = y_pred[7]
            if np.sum(policy_calculated) > 0:
                #error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), ERROR_LIMIT)
                error = -np.log(np.sum(output*batch[1], axis=1))
                X,Y,sample_weight = restore_exp3(settings, input_x, error, input_h, policy, policy_calculated, chosen_action, policy_depth)
                loss2 = rl_model.evaluate(X,Y, batch_size=settings['batch_size'], verbose=0)
                loss2_total.append(loss2)
            depth_total.append(depth[0])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])
            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                             .format(i+1, val_epoch_size,
                                     np.sum(loss1_total)*1.0/len(loss1_total),
                                     np.sum(acc_total)*1.0/len(acc_total),
                                     np.sum(loss2_total)*1.0/len(loss2_total),
                                     np.sum(depth_total)*1.0/len(depth_total)))

def restore_exp(settings, x, total_error, h, policy, fk_calculated, chosen_action, policy_depth):
    max_policy_depth = np.max(policy_depth, axis=(1,2))
    max_policy_depth = np.repeat(np.expand_dims(max_policy_depth, axis=1), policy_depth.shape[1], axis=1)
    max_policy_depth = np.repeat(np.expand_dims(max_policy_depth, axis=2), policy_depth.shape[2], axis=2)
    policy_depth = max_policy_depth - policy_depth
    policy_depth = np.power(settings['rl_gamma'], policy_depth)

    #depth_mult = np.logspace(fk_calculated.shape[1]-1, 0, num=fk_calculated.shape[1], base=settings['rl_gamma'])
    #depth_mult = np.repeat(np.expand_dims(depth_mult, axis=0), fk_calculated.shape[0], axis=0)
    #depth_mult = np.repeat(np.expand_dims(depth_mult, axis=2), fk_calculated.shape[2], axis=2)
    error_mult = np.repeat(np.expand_dims(total_error, axis=1), fk_calculated.shape[1], axis=1)
    error_mult = error_mult# * policy_depth
    error_mult = np.repeat(np.expand_dims(error_mult, axis=2), fk_calculated.shape[2], axis=2)
    #chosen_action = np.less_equal(policy[:,:,:,0], policy[:,:,:,1])
    shift_action_mask = np.ones_like(error_mult)*chosen_action
    reduce_action_mask = np.ones_like(error_mult)*(1-chosen_action)
    shift_action_policy = np.concatenate((np.expand_dims(shift_action_mask*error_mult, axis=3), np.expand_dims(policy[:,:,:,1], axis=3)), axis=3)
    shift_action_policy = np.repeat(np.expand_dims(shift_action_mask, axis=3), 2, axis=3)*shift_action_policy
    reduce_action_policy = np.concatenate((np.expand_dims(policy[:,:,:,0], axis=3), np.expand_dims(reduce_action_mask*error_mult, axis=3)), axis=3)
    reduce_action_policy = np.repeat(np.expand_dims(reduce_action_mask, axis=3), 2, axis=3)*reduce_action_policy
    new_policy = shift_action_policy + reduce_action_policy
    decision_performed = np.where(fk_calculated == 1)
    x_value_input = x[decision_performed]
    h_value_input = h[decision_performed]
    sample_weight = policy_depth[decision_performed]
    policy_output = new_policy[decision_performed]
    return [x_value_input, h_value_input], policy_output, sample_weight


def restore_exp3(settings, x, total_error, h, policy, fk_calculated, chosen_action, policy_depth):
    num_of_decisions = np.sum(fk_calculated, axis=(1,2))
    predicted_error = np.min(policy, axis=-1)
    predicted_error = predicted_error * fk_calculated
    predicted_error = np.sum(predicted_error, axis=(1,2))

    #depth_mult = np.logspace(fk_calculated.shape[1]-1, 0, num=fk_calculated.shape[1], base=settings['rl_gamma'])
    #depth_mult = np.repeat(np.expand_dims(depth_mult, axis=0), fk_calculated.shape[0], axis=0)
    #depth_mult = np.repeat(np.expand_dims(depth_mult, axis=2), fk_calculated.shape[2], axis=2)
    total_error = (total_error - predicted_error) / num_of_decisions

    total_error[total_error == np.inf] = 0
    error_mult = np.repeat(np.expand_dims(total_error, axis=1), fk_calculated.shape[1], axis=1)
    error_mult = error_mult# * policy_depth
    error_mult = np.repeat(np.expand_dims(error_mult, axis=2), fk_calculated.shape[2], axis=2)
    #chosen_action = np.less_equal(policy[:,:,:,0], policy[:,:,:,1])
    shift_action_mask = np.ones_like(error_mult)*chosen_action
    reduce_action_mask = np.ones_like(error_mult)*(1-chosen_action)
    shift_action_policy = np.concatenate((np.expand_dims(shift_action_mask*error_mult, axis=3), np.expand_dims(np.zeros_like(policy[:,:,:,1]), axis=3)), axis=3)
    shift_action_policy = np.repeat(np.expand_dims(shift_action_mask, axis=3), 2, axis=3)*shift_action_policy
    reduce_action_policy = np.concatenate((np.expand_dims(np.zeros_like(policy[:,:,:,0]), axis=3), np.expand_dims(reduce_action_mask*error_mult, axis=3)), axis=3)
    reduce_action_policy = np.repeat(np.expand_dims(reduce_action_mask, axis=3), 2, axis=3)*reduce_action_policy
    policy_update = shift_action_policy + reduce_action_policy
    new_policy = policy+policy_update
    decision_performed = np.where(fk_calculated == 1)
    x_value_input = x[decision_performed]
    h_value_input = h[decision_performed]
    sample_weight = policy_depth[decision_performed]
    policy_output = new_policy[decision_performed]
    return [x_value_input, h_value_input], policy_output, sample_weight


def restore_exp2(settings, x, total_error, h, policy, fk_calculated, chosen_action, policy_depth):
    max_policy = np.max(policy, axis=3)
    next_step_max_policy_reduce_buf = max_policy[:,1:,:]*chosen_action[:,:-1,:]
    next_step_max_policy_reduce = np.zeros_like(max_policy)
    next_step_max_policy_reduce[:,:-1,:] = next_step_max_policy_reduce_buf

    next_step_max_policy_shift_buf = max_policy[:,:,1:]*(1-chosen_action)[:,:,:-1]
    next_step_max_policy_shift = np.zeros_like(max_policy)
    next_step_max_policy_shift[:,:,:-1] = next_step_max_policy_shift_buf
    next_step_max_policy = next_step_max_policy_shift + next_step_max_policy_reduce

    max_depth = np.max(policy_depth, axis=(1,2))
    max_depth = np.repeat(np.expand_dims(max_depth, axis=1), policy_depth.shape[1], axis=1)
    max_depth = np.repeat(np.expand_dims(max_depth, axis=2), policy_depth.shape[2], axis=2)

    last_policy_calculated = (policy_depth == max_depth)*fk_calculated


    error_mult = np.repeat(np.expand_dims(total_error, axis=1), fk_calculated.shape[1], axis=1)
    error_mult = error_mult# * policy_depth
    error_mult = np.repeat(np.expand_dims(error_mult, axis=2), fk_calculated.shape[2], axis=2)
    reward = last_policy_calculated * error_mult


    total_reward = next_step_max_policy*settings['rl_gamma']+reward
    error_mult = total_reward

    shift_action_mask = np.ones_like(error_mult)*chosen_action
    reduce_action_mask = np.ones_like(error_mult)*(1-chosen_action)
    shift_action_policy = np.concatenate((np.expand_dims(shift_action_mask*error_mult, axis=3), np.expand_dims(policy[:,:,:,1], axis=3)), axis=3)
    shift_action_policy = np.repeat(np.expand_dims(shift_action_mask, axis=3), 2, axis=3)*shift_action_policy
    reduce_action_policy = np.concatenate((np.expand_dims(policy[:,:,:,0], axis=3), np.expand_dims(reduce_action_mask*error_mult, axis=3)), axis=3)
    reduce_action_policy = np.repeat(np.expand_dims(reduce_action_mask, axis=3), 2, axis=3)*reduce_action_policy
    new_policy = shift_action_policy + reduce_action_policy
    decision_performed = np.where(fk_calculated == 1)
    x_value_input = x[decision_performed]
    h_value_input = h[decision_performed]
    sample_weight = policy_depth[decision_performed]
    policy_output = new_policy[decision_performed]
    return [x_value_input, h_value_input], policy_output, sample_weight