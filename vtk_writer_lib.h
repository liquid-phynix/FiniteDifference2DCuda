#include<cstdio>
#include<cstdlib>

template<typename f>
int writeImageData(char* fileName, int dimRow, int dimCol, f* dataPointer, bool skipEdges){
  char vtkFormatStringPart1[] =
    "<?xml version=\"1.0\"?>\n"
    "<VTKFile type=\"ImageData\"\n"
    "         version=\"0.1\"\n"
    "         byte_order=\"LittleEndian\">\n"
    "         <ImageData WholeExtent=\"";
  char vtkFormatStringPart2[] =
    " 0 0\"\n"
    "                    Origin=\"0.0 0.0 0.0\"\n"
    "                    Spacing=\"1.0 1.0 0.0\">\n"
    "                    <Piece Extent=\"";
  char vtkFormatStringPart3[] =
    " 0 0\">\n"
    "                           <CellData Scalars=\"field\">\n"
    "                                      <DataArray Name=\"field\"\n"
    "                                                 type=\"Float";
  char vtkFormatStringPart4[] = "\"\n"
    "                                                 format=\"appended\"\n"
    "                                                 offset=\"0\"/>\n"
    "                           </CellData>\n"
    "                    </Piece>\n"
    "         </ImageData>\n"
    "         <AppendedData encoding=\"raw\">\n";
  char vtkFormatStringPart5[] =
    "         </AppendedData>\n"
    "</VTKFile>\n";
  
  FILE* output;
  int dimRealRow = (skipEdges ? (dimRow - 2) : dimRow);
  int dimRealCol = (skipEdges ? (dimCol - 2) : dimCol);
  int arrayLengthInBytes = dimRealRow * dimRealCol * sizeof(f);
  output = fopen(fileName, "w");
  output || (exit(1),0);
  fprintf(output, vtkFormatStringPart1);
  fprintf(output, "%d %d %d %d", 0, dimRealRow, 0, dimRealCol);
  fprintf(output, vtkFormatStringPart2);
  fprintf(output, "%d %d %d %d", 0, dimRealRow, 0, dimRealCol);
  fprintf(output, vtkFormatStringPart3);
  fprintf(output, "%d", 8 * sizeof(f));
  fprintf(output, vtkFormatStringPart4);
  fprintf(output, "_");
  fwrite(&arrayLengthInBytes, sizeof(int), 1, output);
  if(skipEdges){
    int row, offset;
    for(row = 1, offset = dimRow + 1; row < dimCol - 1; row++, offset += dimRow){
      fwrite(dataPointer + offset, sizeof(f), dimRealRow, output);
    }
  }
  else{
    fwrite(dataPointer, sizeof(f), dimRealRow * dimRealCol, output);
  }
  fprintf(output, "\n");
  fprintf(output, vtkFormatStringPart5);
  fclose(output);
  return 0;
}
