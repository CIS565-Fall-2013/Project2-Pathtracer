//--------------------------------------------------------------------------------
///	\file	objload.h
///	\author	Rob Bateman [mailto:robthebloke@hotmail.com] [http://robthebloke.org]
///	\date	11-4-05
///	\brief	A C++ loader for the alias wavefront obj file format. This loader is
///			intended to be as complete as possible (ie, it handles everything that 
///			Maya can spit at it, as well as all the parametric curve and surface 
///			stuff not supported by Maya's objexport plugin). 
///
///			If you want to add obj import/export capabilities to your applications,
///			the best method is probably to derive a class from Obj::File and then
///			use the data in the arrays provided (or fill the data arrays with the 
///			correct info, then export).
/// 
///			In addition to simply accessing the data arrays, you can also overload
///			the functions, PreSave,PostSave,PreLoad,PostLoad and HandleUnknown to 
///			extend the capabilities of the file format (if you really think that 
///			is wise!). 
/// 
///			Generally speaking, the obj file format is actually very very complex.
///			As a result, not all of the Curve and Surface types are fully supported
///			in the openGL display parts of this code. Bezier, Bspline & NURBS curves
///			and surface should be fine. Cardinal, taylor and basis matrix types 
///			will be displayed as CV's / HULLS only. Trim curves and holes don't get
///			displayed either. I'll leave that as an excersice for the reader. 
/// 
///			The material lib files (*.mtl) are also supported and parsed, this 
///			includes a user extensible interface to load the image files as textures.
///			See the texture manager example for more info. 
/// 
///			One nice aspect of this code is the ability to calculate normals for 
///			the polygonal surfaces, and the ability to convert the objfile into 
///			surfaces and /or vertex arrays. Hopefully it may be useful to someone.
/// 
///			If you wish to use this code within a game, I **STRONGLY** RECOMMEND 
///			converting the data to Vertex Arrays and saving as your own custom 
///			file format. The code to handle your own objects will be substantially 
///			more efficient than this code which is designed for completeness, not
///			speed! See the obj->game converter for a quick example of this. 
/// \todo	
///			
///
//--------------------------------------------------------------------------------

#ifndef __OBJ_LOAD__H__
#define __OBJ_LOAD__H__

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif
#include <GL/gl.h>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <iostream>
//#include <Debug.h>

#define FCOMPARE(x,y) (((x)-0.0001f)<(y) && ((x)+0.00001f)>(y))

// debugging macro from other project....
#define MF_FUNC(x) ;


namespace Obj {

	// forward declaration
	class File;

	//-----------------------------------------------------		Obj :: Vertex
	struct Vertex {
		/// x component of point
		float x;
		/// y component of point
		float y;
		/// z component of point
		float z;
		/// ctor from stream
		Vertex(std::istream& ifs) {
			ifs >> x >> y >> z;
		}
		/// ctor
		Vertex() 
			: x(0),y(0),z(0) {
		}
		/// copy ctor
		Vertex(const Vertex& v) 
			: x(v.x),y(v.y),z(v.z) {
		}
		/// stream insertion operator
		friend std::ostream& operator << (std::ostream& ofs,const Vertex& v) {
			return ofs << "v " << v.x << " " << v.y << " " << v.z << std::endl;
		}
		/// calls glVertex
		inline void gl() const {
			glVertex3f(x,y,z);
		}
		bool operator==(const Vertex& v) const {
			return FCOMPARE(x,v.x) && FCOMPARE(y,v.y) && FCOMPARE(z,v.z);
		}
	};

	//-----------------------------------------------------		Obj :: VertexParam
	struct VertexParam {
		/// x component of point
		float u;
		/// y component of point
		float v;
		/// z component of point
		float w;
		/// ctor from stream
		VertexParam(std::istream& ifs) {
			char buffer[128];
			ifs.getline(buffer,128);
			char*ptr = buffer;
			while(*ptr==' '||*ptr=='\t')
				++ptr;
			char*ptr2 = ptr;
			int n=1;
			while (*ptr2!='\0') {
				if (n==' '||n=='\t') {
					++n;
				}
				++ptr2;
			}
			switch(n) {
			case 2:
				sscanf(buffer,"%f%f",&u,&u);
				w=0;
				break;
			case 3:
				sscanf(buffer,"%f%f%f",&u,&v,&w);
				break;
			default:
				break;
			}
		}
		/// ctor
		VertexParam() 
			: u(0),v(0),w(0) {
		}
		/// copy ctor
		VertexParam(const VertexParam& v_) 
			: u(v_.u),v(v_.v),w(v_.w) {
		}
		/// stream insertion operator
		friend std::ostream& operator << (std::ostream& ofs,const VertexParam& v_) {
			return ofs << "vp " << v_.u << " " << v_.v << " " << v_.w << std::endl;
		}
		bool operator==(const VertexParam& v_) const {
			return FCOMPARE(u,v_.u) && 
				   FCOMPARE(v,v_.v) && 
				   FCOMPARE(w,v_.w) ;
		}
	};

	//-----------------------------------------------------		Obj :: Normal
	struct Normal {
		/// x component of vector
		float x;
		/// y component of vector
		float y;
		/// z component of vector
		float z;
		/// ctor from input stream
		Normal(std::istream& ifs) {
			ifs >> x >> y >> z;
		}
		/// ctor
		Normal() 
			: x(0),y(0),z(0) {
		}
		/// copy ctor
		Normal(const Normal& n) 
			: x(n.x),y(n.y),z(n.z) {
		}
		/// stream insertion operator
		friend std::ostream& operator << (std::ostream& ofs,const Normal& n) {
			return ofs << "vn " << n.x << " " << n.y << " " << n.z << std::endl;
		}
		/// calls glNormal
		inline void gl() const {
			glNormal3f(x,y,z);
		}
		/// equivalence operator.
		bool operator==(const Normal& v) const {
			return FCOMPARE(x,v.x) && FCOMPARE(y,v.y) && FCOMPARE(z,v.z);
		}
	};

	//-----------------------------------------------------		Obj :: TexCoord
	struct TexCoord {
		/// u tex coord
		float u;
		/// v tex coord
		float v;
		/// ctor from stream
		TexCoord(std::istream& ifs) {
			ifs >> u >> v;
		}
		/// ctor
		TexCoord()
			: u(0),v(0) {
		}
		/// copy ctor
		TexCoord(const TexCoord& uv) 
			: u(uv.u),v(uv.v) {
		}
		/// stream insertion operator
		friend std::ostream& operator << (std::ostream& ofs,const TexCoord& uv) {
			return ofs << "vt " << uv.u << " " << uv.v << std::endl;
		}
		/// calls glTexCoord2f
		inline void gl() const {
			glTexCoord2f(u,v);
		}
		bool operator==(const TexCoord& uv) const {
			return FCOMPARE(u,uv.u) && FCOMPARE(v,uv.v);
		}
	};

	//-----------------------------------------------------		Obj :: Line
	struct Line {

		/// ctor
		Line() 
			: m_Vertices(), 
			  m_TexCoords()
		{}
		/// copy ctor
		Line(const Line& l)
			: m_Vertices(l.m_Vertices), 
			  m_TexCoords(l.m_TexCoords) 
		{}

		/// the material applied to this line
		unsigned short m_Material;
		/// the group to which this line belongs
		unsigned short m_Group;

		/// the vertex indices for the line
		std::vector<unsigned> m_Vertices;
		/// the texture coord indices for the line
		std::vector<unsigned> m_TexCoords;
		/// output stream operator for line (for writing obj files)
		friend std::ostream& operator << (std::ostream& ofs,const Line& l) {

			ofs << "l";
			if(l.m_TexCoords.size()) {
				std::vector<unsigned>::const_iterator itv= l.m_Vertices.begin();
				std::vector<unsigned>::const_iterator itt= l.m_TexCoords.begin();
				for( ; itv != l.m_Vertices.end();++itv) {
					ofs << " " << *itv << "/" << *itt;
				}
				ofs << "\n";
			}
			else {
				std::vector<unsigned>::const_iterator itv= l.m_Vertices.begin();
				for( ; itv != l.m_Vertices.end();++itv) {
					ofs << " " << *itv ;
				}
				ofs << "\n";
			}
			return ofs;
		}
	};

	//-----------------------------------------------------		Obj :: Face
	struct Face {
		/// ctor
		Face() {
			v[0] = v[1] = v[2] =  0;
			// -1 indicates not used
			n[0] = n[1] = n[2] = -1;
			t[0] = t[1] = t[2] = -1;
		}

		/// vertex indices for the triangle
		unsigned v[3];
		/// normal indices for the triangle
		int n[3];
		/// texture coordinate indices for the triangle
		int t[3];	
		/// stream insertion operator
		friend std::ostream& operator << (std::ostream& ofs,const Face& f) ;
	};

	//-----------------------------------------------------		Obj :: Material
	struct Material 
	{
	public:

		/// ctor
		Material();

		/// dtor
		~Material();


		/// copy ctor
		Material(const Material& mat);

		/// applied the material
		void gl() const;

		/// stream insertion operator
		friend std::ostream& operator << (std::ostream& ofs,const Material& f) ;

	public:

		/// material name
		std::string name;

		/// don't know :| Seems to always be 4
		int illum;

		/// ambient
		mutable float Ka[4];
		/// diffuse
		mutable float Kd[4];
		/// specular
		mutable float Ks[4];
		/// transparency
		float Tf[3];
		/// intensity
		float Ni;
		/// specular power
		float Ns;

		/// ambient texture map
		std::string map_Ka;
		/// diffuse texture map
		std::string map_Kd;
		/// specular texture map
		std::string map_Ks;
		/// bump texture map
		std::string map_Bump;

		/// ambient texture object
		unsigned gltex_Ka;
		/// diffuse texture object
		unsigned gltex_Kd;
		/// specular texture object
		unsigned gltex_Ks;
		/// bump map texture object.
		unsigned gltex_Bump;

		/// bump map depth. Only used if bump is relevent.
		float Bm;
	};

	//-----------------------------------------------------		Obj :: MaterialGroup
	struct MaterialGroup {
		MaterialGroup() {
			m_MaterialIdx = m_StartFace = m_EndFace = 0;
		}
		MaterialGroup(const MaterialGroup& mg) 
			: m_MaterialIdx(mg.m_MaterialIdx), m_StartFace(mg.m_StartFace), m_EndFace(mg.m_EndFace) {}

		/// the material applied to a set of faces
		unsigned m_MaterialIdx;
		/// the starting index of the face to which the material is applied
		unsigned m_StartFace;
		/// the ending index of the face to which the material is applied
		unsigned m_EndFace;
		/// start index for points to which the material is applied
		unsigned StartPoint;
		/// end index for points to which the material is applied
		unsigned EndPoint;
	};

	//-----------------------------------------------------		Obj :: Group
	struct Group
	{
		/// ctor
		Group() 
			: StartFace(0),EndFace(0),name(""),m_AssignedMaterials()
		{}
		/// copy ctor
		Group(const Group& g) 
			: StartFace(g.StartFace),EndFace(g.EndFace),name(g.name),m_AssignedMaterials(g.m_AssignedMaterials)
		{}
		/// start index for faces in the group (surface)
		unsigned StartFace;
		/// end index for faces in the group (surface)
		unsigned EndFace;
		/// start index for points in the group (surface)
		unsigned StartPoint;
		/// end index for points in the group (surface)
		unsigned EndPoint;
		/// name of the group
		std::string name;

		/// a set of material groupings within this surface. ie, which 
		/// materials are assigned to which faces within this group.
		std::vector<MaterialGroup> m_AssignedMaterials;
	};

	//-----------------------------------------------------		Obj :: BezierPatch
	struct BezierPatch {
	public:
		
		/// ctor
		BezierPatch() {
			memset(this,0,sizeof(BezierPatch));
			SetLod(10);
		}

		/// copy ctor
		BezierPatch(const BezierPatch& bzp) {
			// copy over all indices
			for(int i=0;i!=4;++i)
				for(int j=0;j!=4;++j) 
					VertexIndices[i][j] = bzp.VertexIndices[i][j];
			
			/// prevents SetLOD() from attempting to delete invalid data
			IndexData=0;
			VertexData=0;
			BlendFuncs=0;

			/// set level of detail of surface
			SetLod(bzp.LOD);
		}
		
		/// dtor
		~BezierPatch() {
			delete [] IndexData;
			delete [] VertexData;
			delete [] BlendFuncs;
			IndexData=0;
			VertexData=0;
			BlendFuncs=0;
		}

		/// tesselates the surface at the current level of detail
		void Tesselate(const std::vector<Vertex>& point_array);

		/// sets the level of detail and does a bit of internal caching to 
		/// speed things up a little. 
		void SetLod(unsigned new_lod);

		/// renders the surface
		void gl() const;


		friend std::ostream& operator << (std::ostream& ofs,const BezierPatch& bzp) {
			ofs << "bzp ";
			for(int i=0;i!=4;++i) 
				for(int j=0;j!=4;++j)
					ofs << " " << bzp.VertexIndices[i][j];
			return ofs << "\n";
		}

	public:

		/// the material applied to this patch
		unsigned short m_Material;
		/// the group to which this patch belongs
		unsigned short m_Group;

		/// a set of 16 vertex indices
		int VertexIndices[4][4];
		/// an array of vertices/normals/texcoords. Each vertex has 8 floats
		float* VertexData;
		/// an array of vertices/normals/texcoords. Each vertex has 8 floats
		float* BlendFuncs;
		/// an array of vertex indices for triangle strips
		unsigned* IndexData;
		/// the level of detail.
		unsigned LOD;
	};

	//-----------------------------------------------------		Obj :: Surface
	/// the obj file can be split into seperate surfaces 
	/// where all indices are relative to the data in the surface,
	/// rather than all data in the obj file.
	struct Surface {
		friend class File ;
	public:

		/// ctor
		Surface();
		
		/// copy ctor
		Surface(const Surface& surface) ;

		/// function to render the surface
		void gl() const;
		
		/// this function will generate vertex normals for the current 
		/// surface and store those within the m_Normals array
		void CalculateNormals();

	public:

		/// the name of the surface
		std::string name;

		/// the vertices in the obj file
		std::vector<Vertex>   m_Vertices;
		/// the normals from the obj file
		std::vector<Normal>   m_Normals;
		/// the tex coords from the obj file
		std::vector<TexCoord> m_TexCoords;
		/// the triangles in the obj file
		std::vector<Face>     m_Triangles;
		/// the lines in the obj file
		std::vector<Line>     m_Lines;
		/// the points in the obj file
		std::vector<unsigned> m_Points;
		/// a set of material groupings within this surface. ie, which 
		/// materials are assigned to which faces within this group.
		std::vector<MaterialGroup> m_AssignedMaterials;

		
	private:
		/// utility function to draw a range of triangles
		void DrawRange(unsigned start,unsigned end_face) const;
		/// pointer to file to access material data
		File* m_pFile;
	};

	struct GL_Line {
		struct {
			unsigned int numVerts:16;
			unsigned int hasUvs:1;
			unsigned int material:15;
		};
		/// the line indices in the obj file
		std::vector<unsigned int> m_Indices;
	};

	//-----------------------------------------------------		Obj :: VertexBuffer
	/// The obj file can be split into seperate vertex arrays,
	/// ie each group is turned into a surface which uses a 
	/// single index per vertex rather than seperate vertex,
	/// normal and uv indices.
	struct VertexBuffer {
		friend class File ;
	public:

		/// ctor
		VertexBuffer();
		
		/// copy ctor
		VertexBuffer(const VertexBuffer& surface) ;
		
		/// this function will generate vertex normals for the current 
		/// surface and store those within the m_Normals array
		void CalculateNormals();
		
		/// a function to render the vertex array
		void gl() const ;
		
	public:

		/// the name of the surface
		std::string name;

		/// the vertices in the obj file
		std::vector<Vertex>   m_Vertices;
		/// the normals from the obj file
		std::vector<Normal>   m_Normals;
		/// the tex coords from the obj file
		std::vector<TexCoord> m_TexCoords;
		/// the triangles in the obj file
		std::vector<unsigned int> m_Indices;
		/// a set of material groupings within this surface. ie, which 
		/// materials are assigned to which faces within this group.
		std::vector<MaterialGroup> m_AssignedMaterials;
		/// the lines in the obj file.
		std::vector<GL_Line> m_Lines;

	private:
		/// pointer to file to access material data
		File* m_pFile;
	};

	//-----------------------------------------------------		Obj :: File
	/// main interface for an alias wavefront obj file.
	///
	class File {
		// vertex buffer needs to access custom materials
		friend struct VertexBuffer;
		// surfaces may also need to access custom materials if split.
		friend struct Surface;
	public:

		File() {
		}
		virtual ~File() {
			MF_FUNC(File__dtor);
			 Release();
		}

		/// releases all object data
		void Release();

		/// loads the specified obj file
		bool Load(const char filename[]) ;

		/// loads the specified material file
		bool LoadMtl(const char filename[]);

		/// saves the obj file & related material file
		bool Save(const char filename[]) const;

		/// saves the mtl file
		bool SaveMtl(const char filename[]) const;

		/// renders the obj file rather innefficiently using openGL's 
		/// immediate mode
		void Draw() const ;

		/// this function will generate vertex normals for the current 
		/// surface and store those within the m_Normals array
		void CalculateNormals();

		/// splits the obj file into seperate surfaces based upon object grouping.
		/// The returned list of surfaces will use indices relative to the start of
		/// this surface.
		void GroupsToSurfaces(std::vector<Surface>& surface_list);

		/// splits the obj file into sets of vertex arrays for quick rendering.
		void GroupsToVertexArrays(std::vector<VertexBuffer>& surface_list);
	
		std::vector<Face>        m_Triangles;
		/// the groups in the obj file, moved from protected to public
		
		std::vector<Normal>      m_Normals;
		/// the normals from the obj file, moved from protected to public

		std::vector<Vertex>      m_Vertices;
		/// the vertices in the obj file, moved from protected to public


	protected:

		/// overide to handle loading of texture data
		virtual unsigned int OnLoadTexture(const char filename[]) {
			return 0;
		}

		/// overload this to change the way the material is specified 
		/// (ie, custom shaders rather than glMaterialfv() )
		virtual void OnBindMaterial(const Material& mat) const {
			mat.gl();
		}

		/// overload this to perform your own set up before or after saving 
		/// or loading.
		virtual bool PreLoad(std::istream&) {
			return true;
		};
		/// overload this to perform your own set up before or after saving 
		/// or loading.
		virtual bool PostLoad(std::istream&) {
			return true;
		};
		/// overload this to perform your own set up before or after saving 
		/// or loading.
		virtual bool PreSave(std::ostream&) const {
			return true;
		};
		/// overload this to perform your own set up before or after saving 
		/// or loading.
		virtual bool PostSave(std::ostream&) const {
			return true;
		};

		/// overload this to handle additional data types within the obj file 
		/// that I currently do not support.
		virtual bool HandleUnknown(std::string flag,std::istream& ifs) {
			while( !ifs.eof() && (ifs.get()!='\n') )
				/*empty*/ ;
			return true;
		};

	protected:

		/// the vertices in the obj file
		std::vector<VertexParam> m_VertexParams;
		/// the tex coords from the obj file
		std::vector<TexCoord>    m_TexCoords;
		/// the points stored in the obj file
		std::vector<unsigned>    m_Points;
		/// an array of lines
		std::vector<Line>	     m_Lines;
		/// the triangles in the obj file
		std::vector<Group>       m_Groups;
		/// the materials from the mtl file exported from maya
		std::vector<Material>    m_Materials;
		/// an array of bezier patches
		std::vector<BezierPatch> m_Patches;
		/// an array of parametric curves
		/// an array of parametric surfaces

	private:

		/// the name of the file last loaded is saved internally since the 
		/// name may be needed to find a path to a material or texture file.
		std::string m_FilePath;

		std::string ReadChunk(std::istream& ifs) ;

		/// utility function to parse a face 
		void ReadPoints(std::istream&);

		/// utility function to parse a face 
		void ReadLine(std::istream&);

		/// utility function to parse a face 
		void ReadFace(std::istream&);

		/// a utility function to parse a group
		void ReadGroup(std::istream&);

		/// a utility function to draw the specified range of triangles
		void DrawRange(unsigned int start_face,unsigned int end_face) const;

	};
}

#endif
