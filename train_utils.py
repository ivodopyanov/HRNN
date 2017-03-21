# -*- coding: utf-8 -*-
import sys
import numpy as np
import keras.backend as K


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
    predictor.get_layer('encoder').gammas.set_value(K.get_value(encoder.get_layer('encoder').gammas))
    predictor.get_layer('encoder').betas.set_value(K.get_value(encoder.get_layer('encoder').betas))
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
    predictor.get_layer('encoder').W_action_2.set_value(K.get_value(rl_model.get_layer('encoder').W_action_2))
    predictor.get_layer('encoder').b_action_2.set_value(K.get_value(rl_model.get_layer('encoder').b_action_2))

def copy_weights_rl_to_encoder(objects):
    encoder = objects['encoder']
    rl_model = objects['rl_model']
    encoder.get_layer('encoder').W_action_1.set_value(K.get_value(rl_model.get_layer('encoder').W_action_1))
    encoder.get_layer('encoder').U_action_1.set_value(K.get_value(rl_model.get_layer('encoder').U_action_1))
    encoder.get_layer('encoder').b_action_1.set_value(K.get_value(rl_model.get_layer('encoder').b_action_1))
    encoder.get_layer('encoder').W_action_2.set_value(K.get_value(rl_model.get_layer('encoder').W_action_2))
    encoder.get_layer('encoder').b_action_2.set_value(K.get_value(rl_model.get_layer('encoder').b_action_2))



def run_training(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(1*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))

    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))

    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss1_total = []
        acc_total = []
        loss2_total = []
        depth_total = []
        for j in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = encoder.train_on_batch(batch[0], batch[1])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])

            if len(loss1_total) == 0:
                avg_loss1 = 0
            else:
                avg_loss1 = np.sum(loss1_total)/len(loss1_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)/len(acc_total)

            if settings['depth'] > 1:
                if len(loss2_total) == 0:
                    avg_loss2 = 0
                else:
                    avg_loss2 = np.sum(loss2_total)/len(loss2_total)
                if len(depth_total) == 0:
                    avg_depth = 0
                else:
                    avg_depth = np.sum(depth_total)/len(depth_total)

            if settings['mode'] == 0:
                sys.stdout.write("\r batch {} / {}: loss1 = {:.4f}, acc = {:.4f}"
                         .format(j+1, epoch_size, avg_loss1, avg_acc))
            else:
                sys.stdout.write("\r batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                         .format(j+1, epoch_size,
                                 avg_loss1, avg_acc, avg_loss2, avg_depth))

        settings['copy_etp'](objects)
        if settings['mode'] == 1:
            for j in range(epoch_size):
                batch = next(objects['data_gen'])
                y_pred = predictor.predict_on_batch(batch[0])

                output = y_pred[0]
                action = y_pred[1]
                action_calculated = y_pred[2]
                x = y_pred[3]
                h = y_pred[4]
                policy = y_pred[5]
                depth = y_pred[6]

                error = np.sum(output*batch[1], axis=1)
                X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
                loss2 = rl_model.train_on_batch(X,Y)

                loss2_total.append(loss2)
                depth_total.append(depth[0])

                copy_weights_rl_to_predictor(objects)

                if len(loss1_total) == 0:
                    avg_loss1 = 0
                else:
                    avg_loss1 = np.sum(loss1_total)/len(loss1_total)
                if len(acc_total) == 0:
                    avg_acc = 0
                else:
                    avg_acc = np.sum(acc_total)/len(acc_total)
                if len(loss2_total) == 0:
                    avg_loss2 = 0
                else:
                    avg_loss2 = np.sum(loss2_total)/len(loss2_total)
                if len(depth_total) == 0:
                    avg_depth = 0
                else:
                    avg_depth = np.sum(depth_total)/len(depth_total)

                sys.stdout.write("\r batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                             .format(j+1, epoch_size,
                                     avg_loss1, avg_acc, avg_loss2, avg_depth))
            sys.stdout.write("\n")
            copy_weights_rl_to_encoder(objects)

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
            action = y_pred[1]
            action_calculated = y_pred[2]
            x = y_pred[3]
            h = y_pred[4]
            policy = y_pred[5]
            depth = y_pred[6]
            error = -np.log(np.sum(output*batch[1], axis=1))
            X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
            loss2 = rl_model.evaluate(X,Y, batch_size=settings['batch_size'], verbose=0)

            loss2_total.append(loss2)
            depth_total.append(depth[0])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])
            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                             .format(i+1, val_epoch_size,
                                     np.sum(loss1_total)/len(loss1_total),
                                     np.sum(acc_total)/len(acc_total),
                                     np.sum(loss2_total)/len(loss2_total),
                                     np.sum(depth_total)/len(depth_total)))




def run_training2(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(1*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))

    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))

    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss1_total = []
        acc_total = []
        loss2_total = []
        depth_total = []
        for j in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = encoder.train_on_batch(batch[0], batch[1])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])

            settings['copy_etp'](objects)

            y_pred = predictor.predict_on_batch(batch[0])

            output = y_pred[0]
            action = y_pred[1]
            action_calculated = y_pred[2]
            x = y_pred[3]
            h = y_pred[4]
            policy = y_pred[5]
            depth = y_pred[6]

            error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), 1000)
            X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
            loss2 = rl_model.train_on_batch(X,Y)

            loss2_total.append(loss2)
            depth_total.append(depth[0])

            copy_weights_rl_to_predictor(objects)

            if len(loss1_total) == 0:
                avg_loss1 = 0
            else:
                avg_loss1 = np.sum(loss1_total)/len(loss1_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)/len(acc_total)
            if len(loss2_total) == 0:
                avg_loss2 = 0
            else:
                avg_loss2 = np.sum(loss2_total)/len(loss2_total)
            if len(depth_total) == 0:
                avg_depth = 0
            else:
                avg_depth = np.sum(depth_total)/len(depth_total)

            copy_weights_rl_to_encoder(objects)

            sys.stdout.write("\r batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                         .format(j+1, epoch_size,
                                 avg_loss1, avg_acc, avg_loss2, avg_depth))


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
            action = y_pred[1]
            action_calculated = y_pred[2]
            x = y_pred[3]
            h = y_pred[4]
            policy = y_pred[5]
            depth = y_pred[6]
            error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), 1000)
            X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
            loss2 = rl_model.evaluate(X,Y, batch_size=settings['batch_size'], verbose=0)

            loss2_total.append(loss2)
            depth_total.append(depth[0])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])
            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                             .format(i+1, val_epoch_size,
                                     np.sum(loss1_total)/len(loss1_total),
                                     np.sum(acc_total)/len(acc_total),
                                     np.sum(loss2_total)/len(loss2_total),
                                     np.sum(depth_total)/len(depth_total)))




def run_training_encoder_only(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(1*settings['batch_size']))
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
                avg_loss1 = np.sum(loss1_total)/len(loss1_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)/len(acc_total)

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
                                     np.sum(loss1_total)/len(loss1_total),
                                     np.sum(acc_total)/len(acc_total)))



def run_training_RL_only(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(1*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))
    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))
    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss2_total = []
        depth_total = []
        for j in range(epoch_size):
            batch = next(objects['data_gen'])
            y_pred = predictor.predict_on_batch(batch[0])
            output = y_pred[0]
            action = y_pred[1]
            action_calculated = y_pred[2]
            x = y_pred[3]
            h = y_pred[4]
            policy = y_pred[5]
            depth = y_pred[6]
            error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), 1000)
            X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
            loss2 = rl_model.train_on_batch(X,Y)
            loss2_total.append(loss2)
            depth_total.append(depth[0])
            copy_weights_rl_to_predictor(objects)
            if len(loss2_total) == 0:
                avg_loss2 = 0
            else:
                avg_loss2 = np.sum(loss2_total)/len(loss2_total)
            if len(depth_total) == 0:
                avg_depth = 0
            else:
                avg_depth = np.sum(depth_total)/len(depth_total)
            sys.stdout.write("\r batch {} / {}: loss2 = {:.4f}, avg depth = {:.2f}".format(j+1, epoch_size, avg_loss2, avg_depth))
            copy_weights_rl_to_encoder(objects)
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
            action = y_pred[1]
            action_calculated = y_pred[2]
            x = y_pred[3]
            h = y_pred[4]
            policy = y_pred[5]
            depth = y_pred[6]
            error = np.minimum(-np.log(np.sum(output*batch[1], axis=1)), 1000)
            X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
            loss2 = rl_model.evaluate(X,Y, batch_size=settings['batch_size'], verbose=0)
            loss2_total.append(loss2)
            depth_total.append(depth[0])
            loss1_total.append(loss1[0])
            acc_total.append(loss1[1])
            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}, loss2 = {:.4f}, avg depth = {:.2f}"
                             .format(i+1, val_epoch_size,
                                     np.sum(loss1_total)/len(loss1_total),
                                     np.sum(acc_total)/len(acc_total),
                                     np.sum(loss2_total)/len(loss2_total),
                                     np.sum(depth_total)/len(depth_total)))

def restore_exp(settings, x, total_error, h, policy, fk_calculated):
    error_mult = np.repeat(np.expand_dims(total_error, axis=1), fk_calculated.shape[1], axis=1)
    error_mult = np.repeat(np.expand_dims(error_mult, axis=2), fk_calculated.shape[2], axis=2)
    chosen_action = np.less_equal(policy[:,:,:,0], policy[:,:,:,1])
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
    policy_output = new_policy[decision_performed]
    return [x_value_input, h_value_input], policy_output