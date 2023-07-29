## Copyright (C) 2023 tony
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} complex_to_modulus (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: tony <tony@MOSS>
## Created: 2023-07-29


function [complex_matrix,modulus_matrix]=complex_to_modulus(input_file, output_file)
    % Task 1: Load the complex matrix from .mtx file manually
    if ~exist(input_file, 'file')
        error('Input file not found.');
    end
    
    fileID = fopen(input_file, 'r');
    if fileID == -1
        error('Error opening input file.');
    end
    
    % Read the header to get matrix size and number of non-zero entries
    header = fgetl(fileID);
    if ~strncmp(header, '%%MatrixMarket matrix coordinate complex general', 52)
        fclose(fileID);
        error('Invalid Matrix Market format or not a complex matrix.');
    end
    
    % Skip comments (if any)
    line = fgetl(fileID);
    while strncmp(line, '%', 1)
        line = fgetl(fileID);
    end

    % Read matrix size and number of non-zero entries
    size_info = sscanf(line, '%d %d %d');
    m = size_info(1);
    n = size_info(2);
    nnz_count = size_info(3);
    
    % Read data (non-zero entries) using a loop
    I = zeros(nnz_count, 1);
    J = zeros(nnz_count, 1);
    V = zeros(nnz_count, 1);
    for k = 1:nnz_count
        data = sscanf(fgetl(fileID), '%d %d %e');
        I(k) = data(1) ;  % Adjust for one-based indexing
        J(k) = data(2) ;  % Adjust for one-based indexing
        V(k) = data(3);
    end
    fclose(fileID);

    % Create complex matrix from data
    complex_matrix = sparse(I, J, V, m, n);

    % Task 2: Compute the modulus (magnitude) of the complex matrix
    modulus_matrix = abs(complex_matrix);

    % Task 3: Store the real matrix to disk in .mtx format
    [I, J, V] = find(modulus_matrix);  % Find non-zero entries

    % Open the output file for writing
    file = fopen(output_file, 'w');
    if file == -1
        error('Error opening output file.');
    end

    % Write header
    %fprintf(file,'%');
    fprintf(file, '%%%%MatrixMarket matrix coordinate real general\n');
    fprintf(file, '%d %d %d\n', m, n, numel(V));

    % Write data (non-zero entries)
    for k = 1:numel(V)
        fprintf(file, '%d %d %.16e\n', I(k), J(k), V(k));
    end

    % Close the output file
    fclose(file);
end

