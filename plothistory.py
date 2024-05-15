import json
import matplotlib.pyplot as plt

def save_plothistory(model):

    history_dict = model.history
    # Save it under the form of a json file
    json.dump(str(history_dict), open("result/history.json", 'w'))
    # accuracy grafigi
    fig=plt.figure()
    plt.plot(history_dict['capsnet_accuracy'])
    plt.plot(history_dict['val_capsnet_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    fig.savefig("result/base_model/train_accuracy.png")

    #loss grafigi
    fig1=plt.figure()
    plt.plot(history_dict['capsnet_loss'])
    plt.plot(history_dict['val_capsnet_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    fig1.savefig("result/base_model/train_loss.png")
    return history_dict


