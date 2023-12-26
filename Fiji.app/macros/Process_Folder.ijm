/*
 * Macro template to process multiple images in a folder
 */

suffix= ".tif"

input = "D:/Downloads/new_spines/12163/Slice/"
output = "D:/Downloads/new_spines/12163/Deconvolved/"

setBatchMode("hide");
processFolder(input);
setBatchMode("show");

//input = "D:/Downloads/complex/12185/Images/"
//output = "D:/Downloads/complex/12185/Deconvolved/"
//
//setBatchMode("hide");
//processFolder(input);
//setBatchMode("show");
//
//input = "D:/Downloads/complex/17030/Images/"
//output = "D:/Downloads/complex/17030/Deconvolved/"
//
//setBatchMode("hide");
//processFolder(input);
//setBatchMode("show");
//
//input = "D:/Downloads/complex/17228/Images/"
//output = "D:/Downloads/complex/17228/Deconvolved/"
//
//setBatchMode("hide");
//processFolder(input);
//setBatchMode("show");

//exit

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
//		if(File.isDirectory(input + File.separator + list[i]))
//			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	print("Processing: " + input + file);
//	open(file);
	open(input + file);
	print(file);
	// name = getTitle;
//    dotIndex = indexOf(file, ".");
//    title = substring(file, 0, dotIndex);
	run("Iterative Deconvolve 3D", "image=" + file + " point=" + file + " output=Deconvolved show perform wiener=0.000 low=1 z_direction=0.3 maximum=5 terminate=0.010");
	selectWindow("Deconvolved_5");
	saveAs(".tiff", output + File.separator + "Deconvolved_5_" + file);
	closeWindow();
	print("Saving to: " + output);
}

function closeWindow(){
	while (nImages>0) { 
          selectImage(nImages); 
          close(); 
      } 
}
