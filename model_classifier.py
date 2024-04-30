## Imports
# basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle, argparse, os
import data_analysis as da
import shutil, traceback

# data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# Hugging Face
import torch
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset


class NetSkopeTitleClassifier():
    # couple of other things to do. we might want to allow saving of tokenizer and binarizer to a local file
    # allow initiating model with loaded tokenizer and binarizer

    def __init__(self, output_folder, model='BertForSequenceClassification'):
        '''
        INITILIASES NLPModelsTitleLabeler Class
        ========================================
        Params
        model: Takes in either 'BertForSequenceClassification' or 'DistilBertForSequenceClassification'
        '''
        self.file_directory = output_folder
        print(self.file_directory)

        self.model = model
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("Device for training: {}".format(self.device))

        self.model_map = {
            'BertForSequenceClassification': {'model_name': 'bert-base-uncased', 'tokenizer': 'BertTokenizer',
                                              'config': 'BertConfig'},
            'DistilBertForSequenceClassification': {'model_name': 'distilbert-base-uncased',
                                                    'tokenizer': 'DistilBertTokenizer', 'config': 'DistilBertConfig'}
        }

        self.need_special_tokens = {'GPT2ForSequenceClassification'}

        assert self.model in self.model_map.keys(), 'Model not found! Please check that the model input is expected'

        # init tokenzier and model
        self.model_name = self.model_map.get(self.model, {}).get('model_name', None)
        dynamic_tokenizer = globals().get(self.model_map.get(self.model, {}).get('tokenizer', None))
        self.tokenizer = dynamic_tokenizer.from_pretrained(self.model_name)

    def prepare_data_for_training(self, path, dropna=True, test_size=0.2, seed=42):
        '''
        path: filepath of the clean dataset for training
        test_size: test_size for train test split
        seed: seed for random state for train test split
        function returns train test set
        '''

        # check that columns are as we expect.
        expected_columns = ['Job Title', 'Job Function', 'Job Role', 'Job Level']
        data = pd.read_csv(path)
        assert set(list(np.intersect1d(data.columns, expected_columns))) == set(expected_columns), \
            "Job Title, Job Function, Job Role, Job Level must exist - please check your data"

        data['Labels'] = data[['Job Function', 'Job Level', 'Job Role']].values.tolist()
        # TODO - add cleaning here
        data['Job Title'] = data['Job Title'].astype(str)
        data['Job Title'] = data['Job Title'].str.lower()

        try:
            data['Job Title'] = data['Job Title'].apply(da.remove_numbers)
        except:
            print("Error in removing numbers from Job Titles")
            print(traceback.format_exc())

        try:
            data['Job Title'] = data['Job Title'].apply(da.remove_punctuation)
        except:
            print("Error in removing punctuations from Job Titles")
            print(traceback.format_exc())

        try:
            data['Job Title'] = data['Job Title'].apply(da.replace_underscore)
        except:
            print("Error in replacing underscores in Job Titles")
            print(traceback.format_exc())

        try:
            data['Job Title'] = data['Job Title'].apply(da.remove_duplicate_space)
        except:
            print("Error in replacing duplicated spaces in Job Titles")
            print(traceback.format_exc())

        if dropna:
            data = data.dropna()

        # train test split
        if test_size == 1:
            self.train_df = None
            self.test_df = data
        elif test_size == 0:
            self.train_df = data
            self.test_df = None
        else:
            self.train_df, self.test_df = train_test_split(data, test_size=test_size, random_state=seed)

    def init_mlb(self):
        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(self.train_df['Labels'])

        print('Dumping MultiLabelBinarizer to mlb_trained.pkl')
        with open('{}/mlb_trained.pkl'.format(self.file_directory), 'wb') as f:
            pickle.dump(mlb, f)

        return mlb, train_labels

    def load_mlb(self):
        with open('{}/mlb_trained.pkl'.format(self.file_directory), 'rb') as f:
            print('Loading MultiLabelBinarizer from mlb_trained.pkl')
            mlb = pickle.load(f)

        return mlb

    def encode_labels(self):
        ## we want to initialise mlb and pickle it when training -- probably an init mlb function
        ## encode labels -- this is for test and prediction

        self.mlb = MultiLabelBinarizer()
        ## need some handling here to -- we probably want to save a copy of mlb somewhere -- find out how to do it.
        self.train_label = self.mlb.fit_transform(self.train_df['Labels'])
        self.test_label = self.mlb.transform(self.test_df['Labels'])

        pass

    def tokenize_data(self, tokenizer, data, max_length=128):
        tokenized_data = tokenizer(data['Job Title'].tolist(), truncation=True, padding=True, return_tensors='pt')
        return tokenized_data

    def convert_to_TensorDataset(self, tokenized_data, labels=None):
        if labels is not None:
            tensorDataset = TensorDataset(tokenized_data['input_ids'].squeeze(),
                                          tokenized_data['attention_mask'].squeeze(), labels)
        else:
            tensorDataset = TensorDataset(tokenized_data['input_ids'].squeeze(),
                                          tokenized_data['attention_mask'].squeeze())

        return tensorDataset

    def custom_collator(self, features):
        input_ids, attention_mask, labels = zip(*features)
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }

    def compute_accuracy(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {"accuracy": (preds == labels).mean()}

    def train(self, data_path, output_dir=None, epoch=3, custom_layer_size=None, train_batch_size=16, warmup_steps=500,
              weight_decay=0.01, learning_rate=5e-5, gradient_accumulation_steps=2, logging_dir='./logs', seed=42,
              test_size=0.2):
        '''
        Trains model by taking in several inputs
        =========================================
        Params
        data_path: file directory for training data
        output_dir: directory for saving model checkpoint files
        epoch: Number of epoch
        train_batch_size: batch_size for training
        warmup_steps: warm up steps for model
        weight_decay: weight decay during training
        logging_dir: directory for logs
        learning_rate: learning rate of model
        gradient_accumulation_steps: gradient accumulation step for model
        '''

        assert data_path, 'Please input file path for data'

        self.prepare_data_for_training(data_path, test_size=test_size, seed=seed)

        mlb, mlb_train_label = self.init_mlb()

        dynamic_model = globals().get(self.model)
        self.modelClass = dynamic_model.from_pretrained(self.model_name, num_labels=len(mlb.classes_))

        if custom_layer_size:
            config = BertConfig.from_json_file("./nn_size/config-{}.json".format(custom_layer_size))
            self.modelClass = dynamic_model(config)
            print(self.modelClass.classifier)

        print('Preprocessing Training Data')
        train_data = self.tokenize_data(self.tokenizer, self.train_df)
        train_labels = torch.tensor(mlb_train_label, dtype=torch.float32)
        train_dataset = self.convert_to_TensorDataset(train_data, labels=train_labels)

        # set up training args
        if not output_dir:
            output_dir = '{}/models/{}'.format(self.file_directory, self.model)

        training_args = TrainingArguments(output_dir=output_dir,
                                          num_train_epochs=epoch,
                                          per_device_train_batch_size=train_batch_size,
                                          warmup_steps=warmup_steps,
                                          weight_decay=weight_decay,
                                          logging_dir=logging_dir,
                                          learning_rate=learning_rate,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          )

        trainer = Trainer(
            model=self.modelClass,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.custom_collator
        )

        trainer.train()
        pass

    def evaluate(self, data_path, checkpoint, batch_size=16, seed=42, test_size=0.2):

        ## load model
        dynamic_config = globals().get(self.model_map.get(self.model, {}).get('config', None))
        config = dynamic_config.from_pretrained(checkpoint)
        mlb = self.load_mlb()

        dynamic_model = globals().get(self.model)
        self.modelEval = dynamic_model.from_pretrained(checkpoint, config=config)

        self.modelEval.to(self.device)
        self.modelEval.eval()

        all_preds = []
        all_labels = []

        batch_size = batch_size

        self.prepare_data_for_training(data_path, test_size=test_size, seed=seed)

        test_data = self.tokenize_data(self.tokenizer, self.test_df)
        test_labels = torch.tensor(mlb.fit_transform(self.test_df['Labels']), dtype=torch.float32)
        test_dataset = self.convert_to_TensorDataset(test_data, labels=test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(test_loader, desc='Evaluating'):
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(
                    self.device), labels.to(self.device)
                outputs = self.modelEval(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()
                preds = (logits > 0).astype(int)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        print("all_preds: ", str(len(all_preds)))
        all_labels = np.concatenate(all_labels, axis=0)
        print("all_labels: ", str(len(all_labels)))
        accuracy = accuracy_score(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, target_names=mlb.classes_)

        print(f'Accuracy: {accuracy:.4f}\n')
        # print('Classification Report:\n', class_report)

        self.test_df.reset_index(drop=True, inplace=True)

        ### Post processing
        evaluation_results = []
        decoded_predictions = mlb.inverse_transform(all_preds)
        post_processed_predictions = self.post_processing(decoded_predictions)
        print("Post Process Predictions Rows: {}".format(len(post_processed_predictions)))
        for i, row in self.test_df.iterrows():
            try:
                title = row['Job Title']
                # row_id = row['Record ID']
                function = row['Job Function']
                role = row['Job Role']
                level = row['Job Level']
                prediction = post_processed_predictions[i]
                pred_function = prediction[0]
                pred_role = prediction[1]
                pred_level = prediction[2]
                pred_is_icp = prediction[3]

                evaluation_results.append(
                    (title, function, role, level, pred_function, pred_role, pred_level, pred_is_icp))
            except:
                print(i, row)
                print(traceback.format_exc())

        evaluation_results_df = pd.DataFrame(evaluation_results,
                                             columns=['Job Title', 'Job Function', 'Job Role', 'Job Level',
                                                      'Job Function Pred', 'Job Role Pred', 'Job Level Pred',
                                                      'Is ICP Pred'])

        # create directory if it doesn't exist
        if os.path.isdir(self.file_directory + "/evaluation_results/") == False:
            print("Creating the folder for storing evaluation results: ", self.file_directory + "/evaluation_results/")
            os.mkdir(self.file_directory + "/evaluation_results")
        else:
            print("Deleting existing Evaluation results and create the folder again")
            shutil.rmtree(self.file_directory + "/evaluation_results/")
            os.mkdir(self.file_directory + "/evaluation_results/")
        csv_output_path = '{}/evaluation_results/evaluation_results-{}.csv'.format(self.file_directory,
                                                                                   checkpoint.split('/')[-1])
        plot_outout_path = '{}/evaluation_results/evaluation_results-{}.png'.format(self.file_directory,
                                                                                    checkpoint.split('/')[-1])

        print('Saving CSV Results to {}'.format(csv_output_path))
        evaluation_results_df.to_csv(csv_output_path, index=False)

        self.plot_evaluation_results(evaluation_results_df, plot_outout_path)

        return class_report, all_preds, all_labels, evaluation_results_df

    def predict(self, checkpoint, file_path=None, title_list=None, output_path=None, batch_size=16):
        # need to write up this function -- takes it either a list or csv path probably.
        assert file_path or title_list, 'Must at least have a csv file or list of titles for prediction'

        levels = {'C-Level', 'Contributor', 'Director', 'Executive', 'Manager'}
        functions = {'IT', 'Engineering', 'Procurement', 'Risk/Legal/Compliance'}
        roles = {'Information Security', 'Networking', 'IT General', 'Development', 'Systems', 'No Sub-Roles'}

        batch_size = batch_size
        mlb = self.load_mlb()

        # prepare data
        if title_list:
            assert isinstance(title_list, list), 'Please input title_list as a list'
            tokenised_title = self.tokenizer(title_list, truncation=True, padding=True, return_tensors='pt')

        else:
            assert '.csv' in file_path[-4:].lower(), 'Input file path must be a csv file'
            expected_columns = ['Record ID', 'Job Title']
            data = pd.read_csv(file_path).astype(str)
            assert all(column in data.columns for column in
                       expected_columns), 'Please ensure that the csv contains ID and Job Title Column'
            data['Original Title'] = data['Job Title']
            # TODO: Add code for cleaning "title" for prediction
            data['Job Title'] = data['Job Title'].astype(str)
            data['Job Title'] = data['Job Title'].str.lower()
            data['Job Title'] = data['Job Title'].apply(da.remove_numbers)
            data['Job Title'] = data['Job Title'].apply(da.remove_punctuation)
            data['Job Title'] = data['Job Title'].apply(da.replace_underscore)
            data['Job Title'] = data['Job Title'].apply(da.remove_duplicate_space)
            data = data.dropna()
            print("---- tokenizing")
            tokenised_title = self.tokenize_data(self.tokenizer, data)
        print("-------------------------------------------- pred_dataset converting")
        pred_dataset = self.convert_to_TensorDataset(tokenised_title)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

        ###### load model ######
        dynamic_config = globals().get(self.model_map.get(self.model, {}).get('config', None))
        config = dynamic_config.from_pretrained(checkpoint)

        dynamic_model = globals().get(self.model)
        self.modelEval = dynamic_model.from_pretrained(checkpoint, config=config)

        self.modelEval.to(self.device)
        self.modelEval.eval()

        all_preds = []

        #### Predict ####
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(pred_loader, desc='Predicting'):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                outputs = self.modelEval(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()
                preds = (logits > 0).astype(int)
                all_preds.append(preds)

        all_preds = np.concatenate(all_preds, axis=0)

        #### Inverse Transform to get label ####
        mlb = self.load_mlb()
        decoded_predictions = mlb.inverse_transform(all_preds)
        # clean up prediction data -- need some processing for predictions for 'OTHERS'
        if title_list:
            results = []
            for i in range(len(title_list)):
                pred_level, pred_function, pred_role = 'Other', 'Other', 'Other'

                title = title_list[i]
                prediction = decoded_predictions[i]
                level_search = set(prediction).intersection(levels)
                function_search = set(prediction).intersection(functions)
                role_search = set(prediction).intersection(roles)

                if len(level_search) == 1: pred_level = list(level_search)[0]
                if len(function_search) == 1: pred_function = list(function_search)[0]
                if len(role_search) == 1: pred_role = list(role_search)[0]

                results.append((title, pred_function, pred_role, pred_level))

            results_df = pd.DataFrame(results, columns=['Job Title', 'Job Function', 'Job Role', 'Job Level'])
        else:
            results = []
            post_processed_predictions = self.post_processing(decoded_predictions)
            for i, row in data.iterrows():
                title = row['Original Title']
                row_id = row['Record ID']
                prediction = post_processed_predictions[i]
                function = prediction[0]
                role = prediction[1]
                level = prediction[2]
                is_icp = prediction[3]

                results.append((row_id, title, function, role, level, is_icp))

            results_df = pd.DataFrame(results,
                                      columns=['Record ID', 'Job Title', 'Job Function', 'Job Role', 'Job Level',
                                               'Is ICP'])

            # TODO - code to check if the folder exists, if not create the folder first
            if os.path.isdir(self.file_directory + "/predicted_values/") == False:
                print("Creating the folder for storing predictions: ", self.file_directory + "/predicted_values/")
                os.mkdir(self.file_directory + "/predicted_values")
            else:
                print("Deleting existing predictions and create the folder again")
                shutil.rmtree(self.file_directory + "/predicted_values/")
                os.mkdir(self.file_directory + "/predicted_values/")
            output_path = '{}/predicted_values/predictions.csv'.format(self.file_directory)

            print('Saving Results to {}'.format(output_path))
            results_df.to_csv(output_path, index=False)

        return results_df

    def post_processing(self, decoded_predictions):
        levels = {'C-Level', 'Contributor', 'Director', 'Executive', 'Manager'}
        functions = {'IT', 'Engineering', 'Procurement', 'Risk/Legal/Compliance'}
        roles = {'Information Security', 'Networking', 'IT General', 'Development', 'Systems'}

        post_processed_predictions = []

        for predictions in decoded_predictions:
            pred_level, pred_function, pred_role = 'Other', 'Other', 'Other'

            level_search = set(predictions).intersection(levels)
            function_search = set(predictions).intersection(functions)
            role_search = set(predictions).intersection(roles)

            if len(level_search) == 1: pred_level = list(level_search)[0]
            if len(function_search) == 1: pred_function = list(function_search)[0]
            if len(role_search) == 1: pred_role = list(role_search)[0]

            if (pred_role in roles) and (pred_function != 'IT'):
                pred_function = "IT"  # update function to be IT if role is IT role

            if (pred_function == 'IT') and (pred_role not in roles):
                pred_role = 'IT General'  # update role to be IT General if function = "IT"

            if (pred_function != 'IT') and (pred_function in functions):
                pred_role = 'Other'

            ### predict ICP or not
            is_ICP = 'ICP'
            if (pred_function == 'Other') and (pred_level in {'Contributor', 'Other', 'Manager'}):
                is_ICP = 'Not ICP'

            post_processed_predictions.append((pred_function, pred_role, pred_level, is_ICP))

        return post_processed_predictions

    def plot_evaluation_results(self, evaluation_results_df, save_path=None):
        columns = {'Job Function', 'Job Role', 'Job Level'}

        ## get accuracies
        overall_acc_dict = {'Category': [], 'Accuracy': []}
        sub_acc_dict = {}

        for column in columns:
            overall_acc = len(
                evaluation_results_df[evaluation_results_df[column] == evaluation_results_df[column + ' Pred']]) / len(
                evaluation_results_df)
            overall_acc_dict['Category'].append(column)
            overall_acc_dict['Accuracy'].append(overall_acc)
            sub_acc_dict[column] = {'Category': [], 'Accuracy': []}

            unique_values = evaluation_results_df[column].unique()
            for unique_key in unique_values:
                sub_df = evaluation_results_df[evaluation_results_df[column] == unique_key]

                sub_acc = len(sub_df[sub_df[column] == sub_df[column + ' Pred']]) / len(sub_df)

                sub_acc_dict[column]['Category'].append(unique_key)
                sub_acc_dict[column]['Accuracy'].append(sub_acc)

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))

        ax = axs[0, 0]
        ax.bar(overall_acc_dict['Category'], overall_acc_dict['Accuracy'], color='darkblue', edgecolor='grey',
               label='Accuracy')
        for i, value in enumerate(overall_acc_dict['Accuracy']):
            ax.text(i, value, str(round(value, 4)), ha='center', va='bottom')
        ax.set_title('Accuracy of Prediction for each overall category')

        colors_dict = {'Job Function': 'skyblue', 'Job Role': 'salmon', 'Job Level': '#C2D69C'}
        subplot_dict = {'Job Function': [1, 0], 'Job Role': [0, 1], 'Job Level': [1, 1]}

        for main_group in sub_acc_dict.keys():

            categories = sub_acc_dict[main_group]['Category']
            acc = sub_acc_dict[main_group]['Accuracy']
            ax = axs[subplot_dict[main_group][0], subplot_dict[main_group][1]]

            ax.bar(categories, acc, color=colors_dict[main_group], edgecolor='grey', label='Accuracy')

            for i, value in enumerate(acc):
                ax.text(i, value + 0.0025, str(round(value, 4)), ha='center', va='bottom')
            ax.set_title('Accuracy of {} Predictions'.format(main_group))

        print('Saving CSV Results to {}'.format(save_path))
        plt.savefig(save_path)

        pass


def main():
    parser = argparse.ArgumentParser(description='NetSkope Title Classifier')
    # init model
    parser.add_argument('-m', '--model', default='BertForSequenceClassification', type=str,
                        help='Model to use -- choose between BertForSequenceClassification or DistilBertForSequenceClassification')
    parser.add_argument('-c', '--checkpoint', default='models/BertForSequenceClassification/checkpoint-12500', type=str,
                        help='Checkpoint file to use')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='Train Model')
    parser.add_argument('-e', '--evaluate', action='store_true', default=False, help='Evaluate Model')
    parser.add_argument('-p', '--predict', action='store_true', default=False, help='Make Prediction')
    parser.add_argument('--data', default=None, type=str, help='Data file to use')

    # shares variables
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size to use')
    parser.add_argument('--seed', default=42, type=int, help='Seed to use for train test split')
    parser.add_argument('--test_size', default=0.2, type=float, help='Test size to use for train test split')

    # train variables
    parser.add_argument('--output_dir', default='models/BertForSequenceClassification', type=str,
                        help='Output directory to store model')
    parser.add_argument('--epochs', default=3, type=int, help='Number of epochs to train')
    parser.add_argument('--warmup_steps', default=500, type=int, help='Number of warmup steps to use')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate to use')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay to use')
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='Gradient accumulation steps to use')
    parser.add_argument('--logging_dir', default='./logs', type=str, help='Logging directory to store logs')
    parser.add_argument('--custom_layer_size', default=None, type=int, help='Custom layer size to use')

    # evaluate variables

    # predict variables
    parser.add_argument('--title_list', default=None, type=str, help='Title list to use for prediction')
    parser.add_argument('--input_file', default=None, type=str, help='Input file to use for prediction')
    parser.add_argument('--output_file', default=None, type=str, help='Output file to use for prediction')

    args = parser.parse_args()

    assert args.predict + args.train + args.evaluate == 1, 'Please only predict, train or evaluate.'

    current_dir = os.getcwd()

    netskope_title_classifier = NetSkopeTitleClassifier(current_dir, args.model)

    if args.train:
        netskope_title_classifier.train(args.data, output_dir=args.output_dir, epoch=args.epochs,
                                        custom_layer_size=args.custom_layer_size, train_batch_size=args.batch_size,
                                        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
                                        learning_rate=args.learning_rate,
                                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                                        logging_dir=args.logging_dir, seed=args.seed, test_size=args.test_size)

    if args.evaluate:
        class_report, all_preds, all_labels, evaluation_results_df = netskope_title_classifier.evaluate(args.data,
                                                                                                        args.checkpoint,
                                                                                                        batch_size=args.batch_size,
                                                                                                        seed=args.seed,
                                                                                                        test_size=args.test_size)
        print(class_report)

    if args.predict:
        netskope_title_classifier.predict(args.checkpoint, file_path=args.input_file, title_list=args.title_list,
                                          output_path=args.output_file)


if __name__ == '__main__':
    main()
