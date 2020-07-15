//
//  ViewController.swift
//  ML with iOS
//
//  Created by kushal dave on 14/07/20.
//  Copyright © 2020 kushal dave. All rights reserved.
//

import UIKit
import Firebase

class ViewController: UIViewController {

    @IBOutlet var tvIdentifiedItem:UILabel!
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // load model from path
        guard let modelPath = Bundle.main.path(
          forResource: "model_unquant",
          ofType: "tflite"
        ) else { return }
        let localModel = CustomLocalModel(modelPath: modelPath)
        //initialise interpreter
        let interpreter = ModelInterpreter.modelInterpreter(localModel: localModel)
        
        
        let ioOptions = ModelInputOutputOptions()
        do {
            try ioOptions.setInputFormat(index: 0, type: .float32, dimensions: [1, 224, 224, 3])
            try ioOptions.setOutputFormat(index: 0, type: .float32, dimensions: [1, 2])
        } catch let error as NSError {
            print("Failed to set input or output format with error: \(error.localizedDescription)")
        }
        
        /* Here we are using static image from drawable to keep the code minimum and avoid distraction,
        Recommended method would be to get the image from user by camera or device photos using
         the same code by handling all this logic in a method and calling that every time */
        
        let image = UIImage(imageLiteralResourceName: "car1").cgImage!
        guard let context = CGContext(
          data: nil,
          width: image.width, height: image.height,
          bitsPerComponent: 8, bytesPerRow: image.width * 4,
          space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
          return
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let imageData = context.data else { return  }

        let inputs = ModelInputs()
        var inputData = Data()
        do {
          for row in 0 ..< 224 {
            for col in 0 ..< 224 {
              let offset = 4 * (col * context.width + row)
              // (Ignore offset 0, the unused alpha channel)
              let red = imageData.load(fromByteOffset: offset+1, as: UInt8.self)
              let green = imageData.load(fromByteOffset: offset+2, as: UInt8.self)
              let blue = imageData.load(fromByteOffset: offset+3, as: UInt8.self)

              // Normalize channel values to [0.0, 1.0]. This requirement varies
              // by model. For example, some models might require values to be
              // normalized to the range [-1.0, 1.0] instead, and others might
              // require fixed-point values or the original bytes.
              var normalizedRed = Float32(red) / 255.0
              var normalizedGreen = Float32(green) / 255.0
              var normalizedBlue = Float32(blue) / 255.0

              // Append normalized values to Data object in RGB order.
              let elementSize = MemoryLayout.size(ofValue: normalizedRed)
              var bytes = [UInt8](repeating: 0, count: elementSize)
              memcpy(&bytes, &normalizedRed, elementSize)
              inputData.append(&bytes, count: elementSize)
              memcpy(&bytes, &normalizedGreen, elementSize)
              inputData.append(&bytes, count: elementSize)
              memcpy(&bytes, &normalizedBlue, elementSize)//changed
              inputData.append(&bytes, count: elementSize)
            }
          }
          try inputs.addInput(inputData)
        } catch let error {
          print("Failed to add input: \(error)")
        }
        
        interpreter.run(inputs: inputs, options: ioOptions) { outputs, error in
            guard error == nil, let outputs = outputs else { return }
            // Process outputs
            let output = try? outputs.output(index: 0) as? [[NSNumber]]
            let probabilities = output?[0]
            //loads labels file
            guard let labelPath = Bundle.main.path(forResource: "labels", ofType: "txt") else { return }
            let fileContents = try? String(contentsOfFile: labelPath)
            guard let labels = fileContents?.components(separatedBy: "\n") else { return }

            var higherProbablityFloat:Float = 0
            for i in 0 ..< labels.count-1 {
              if let probability = probabilities?[i] {
                print("\(labels[i]): \(probability)")
                if (Float(truncating: probability)>higherProbablityFloat) {
                    higherProbablityFloat = Float(truncating: probability)
                    var label = String(labels[i].dropFirst())
                    label = String(label.dropFirst())
                    self.tvIdentifiedItem.text = "The Image is of \(label)"
                }
                
              }
            }
        }
    }
}
