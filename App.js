import React from 'react'
import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  StatusBar,
  ScrollView,
  Image,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native'
import * as tf from '@tensorflow/tfjs'
import { fetch } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as jpeg from 'jpeg-js'
import { ActionSheet, Root } from 'native-base';
import * as ImagePicker from 'expo-image-picker'
import Constants from 'expo-constants'
import * as Permissions from 'expo-permissions'

class App extends React.Component {
  state = {
    isTensorFlowReady: false,
    isModelReady: false,
    predictions: null,
    image: null
  }

  async componentDidMount() {
    await tf.ready()
    this.setState({
      isTensorFlowReady: true
    })
    this.model = await mobilenet.load()
    this.setState({ isModelReady: true })
    this.getPermissionAsync()
  }

  getPermissionAsync = async () => {
    if (Constants.platform.ios) {
      const { status } = await Permissions.askAsync(Permissions.CAMERA_ROLL)
      const { statusCamera } = await Permissions.askAsync(Permissions.CAMERA).status
      if (status !== 'granted') {
        alert('Sorry, we need camera roll permissions to make this work!')
      }
      if (statusCamera !== 'granted') {
        alert('Sorry, we need camera permissions to make this work!')
      }
    }
  }

  imageToTensor(rawImageData) {
    const TO_UINT8ARRAY = true
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY)
    const buffer = new Uint8Array(width * height * 3)
    let offset = 0
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset]
      buffer[i + 1] = data[offset + 1]
      buffer[i + 2] = data[offset + 2]

      offset += 4
    }

    return tf.tensor3d(buffer, [height, width, 3])
  }

  classifyImage = async () => {
    try {
      const imageAssetPath = Image.resolveAssetSource(this.state.image)
      const response = await fetch(imageAssetPath.uri, {}, { isBinary: true })
      const rawImageData = await response.arrayBuffer()
      const imageTensor = this.imageToTensor(rawImageData)
      const predictions = await this.model.classify(imageTensor)
      this.setState({ predictions })
    } catch (error) {
      console.log(error)
    }
  }

  selectImage = async () => {
    try {
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3]
      })

      if (!response.cancelled) {
        const source = { uri: response.uri }
        this.setState({ image: source })
        this.classifyImage()
      }
    } catch (error) {
      console.log(error)
    }
  }

  takePhoto = async () => {
    try {
      let response = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3]
      })
      if (!response.cancelled) {
        const source = { uri: response.uri }
        this.setState({ image: source })
        this.classifyImage()
      }
    } catch (error) {
      console.log(error)
    }
  }

  handleChoosePhoto = (index) => {
    if (index === 0) {
      this.takePhoto();
    } else if (index === 1) {
      this.selectImage();
    }
  }

  renderPrediction = prediction => {
    return (
      <Text key={prediction.className} style={styles.text}>
        {prediction.className}
      </Text>
    )
  }

  renderActionSheet = () => {
    const options = ['Camera', 'Gallery', 'Cancel'];
    ActionSheet.show({
      options,
      cancelButtonIndex: 2,
    }, (buttonIndex) => {
      this.handleChoosePhoto(buttonIndex);
    });
  };

  render() {
    const { isTensorFlowReady, isModelReady, predictions, image } = this.state

    return (
      <Root>
        <SafeAreaView style={styles.container}>
          <ScrollView style={styles.container}>
            <View style={styles.containerTitle}>
            <StatusBar barStyle='light-content' />
            <Text style={styles.title}> Tensor Flow Example </Text>
            <View style={styles.loadingContainer}>
              <View style={{...styles.text, ...styles.textContainer}}>
                <Text style={styles.readyText}>
                  TFJS Ready?
                </Text> 
                <Text>
                  {isTensorFlowReady ? <Text style={{fontSize: 20}}>✅</Text> : ''}
                </Text>
              </View>
              <View style={styles.loadingModelContainer}>
                <View style={{...styles.text, ...styles.textContainer}}>
                <Text style={styles.readyText}>
                  Model Ready?
                </Text>
                {isModelReady ? (
                  <Text style={{fontSize: 20}}>✅</Text>
                ) : (
                  <ActivityIndicator size='small' />
                )}
              </View>
              </View>
            </View>
            
            <TouchableOpacity
              style={styles.imageWrapper}
              onPress={isModelReady ? this.selectImage : undefined}>
              {image && <Image source={image} style={styles.imageContainer} />}

              {isModelReady && !image && (
                <Text style={styles.transparentText}>Tap to choose image</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity style={{marginBottom: 10}} onPress={() => this.renderActionSheet()}>
              <Text style={styles.choosePhotoBtn}> 
                Take a Photo
              </Text>
            </TouchableOpacity>
            <View style={styles.predictionWrapper}>
              {isModelReady && image && (
                <Text style={styles.text}>
                  Predictions: {predictions ? '' : 'Predicting...'}
                </Text>
              )}
              {isModelReady &&
                predictions &&
                predictions.map(p => this.renderPrediction(p))}
            </View>
          </View>
          </ScrollView>
        </SafeAreaView>
      </Root>
    )
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  containerTitle: {
    marginTop: 80,
  },
  choosePhotoBtn: {
    backgroundColor: 'white', 
    width: '70%',
    fontWeight: 'bold',
    alignSelf: 'center', 
    paddingVertical: 20,
    textAlign: 'center',
    borderRadius: 10, 
    marginTop: 10,
    overflow: 'hidden'
  },
  readyText: {
    color: 'white', 
    fontSize: 16, 
    fontWeight: 'bold'
  },
  title: {
    fontSize: 20, 
    fontWeight: 'bold', 
    alignSelf: 'center', 
    color: 'white',
    marginBottom: 30
  },
  textContainer: {
    justifyContent: 'space-between',  
    flexDirection: 'row', 
    width: '90%', 
    alignSelf : 'center'
  },
  text: {
    color: '#ffffff',
    fontSize: 16
  },
  loadingModelContainer: {
    marginTop: 10
  },
  imageWrapper: {
    width: 282,
    height: 280,
    padding: 10,
    borderWidth: 5,
    borderRadius: 10,
    marginTop: 40,
    backgroundColor: 'gray',
    alignSelf: 'center',
    marginBottom: 10,
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center'
  },
  imageContainer: {
    width: 250,
    height: 250,
    position: 'absolute',
    top: 10,
    left: 10,
    bottom: 10,
    right: 10
  },
  predictionWrapper: {
    height: 100,
    width: '100%',
    flexDirection: 'column',
    alignItems: 'center'
  },
  transparentText: {
    color: '#ffffff',
    opacity: 0.7
  },
  footer: {
    marginTop: 40
  },
  poweredBy: {
    fontSize: 20,
    color: '#e69e34',
    marginBottom: 6
  },
  tfLogo: {
    width: 125,
    height: 70
  }
})

export default App
